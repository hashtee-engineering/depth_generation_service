import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Literal
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import threading
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import utils3d

from moge.model import import_model_class_by_version
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal


# --- Configuration ---
class Settings:
    MODEL_VERSION: str = "v2"
    PRETRAINED_MODEL: str = "Ruicheng/moge-2-vitl-normal"
    USE_FP16: bool = True
    MAX_IMAGE_SIZE: int = 800
    RESOLUTION_LEVEL: int = 9
    OUTPUT_DIR: Path = Path(tempfile.gettempdir()) / "moge_outputs"
    FILE_RETENTION_SECONDS: int = 3600  # 1 hour
    MAX_QUEUE_SIZE: int = 100


settings = Settings()
settings.OUTPUT_DIR.mkdir(exist_ok=True)


# --- Job Management ---
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    outputs: dict = field(default_factory=dict)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class JobQueue:
    def __init__(self, max_size: int = 100):
        self.queue: deque = deque()
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.max_size = max_size
        self.condition = threading.Condition(self.lock)

    def add_job(self, job: Job, image: np.ndarray, params: dict) -> bool:
        with self.lock:
            if len(self.queue) >= self.max_size:
                return False
            self.jobs[job.id] = job
            self.queue.append((job.id, image, params))
            self.condition.notify()
            return True

    def get_next(self) -> Optional[tuple]:
        with self.condition:
            while len(self.queue) == 0:
                self.condition.wait()
            return self.queue.popleft()

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        with self.lock:
            if job_id in self.jobs:
                for k, v in kwargs.items():
                    setattr(self.jobs[job_id], k, v)


job_queue = JobQueue(max_size=settings.MAX_QUEUE_SIZE)


# --- Model Loading ---
model = None
model_lock = threading.Lock()


def load_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                print("Loading MoGe model...")
                model = import_model_class_by_version(settings.MODEL_VERSION).from_pretrained(
                    settings.PRETRAINED_MODEL
                ).cuda().eval()
                if settings.USE_FP16:
                    model.half()
                print("Model loaded.")
    return model


# --- Inference Worker ---
def run_inference(image: np.ndarray, resolution_level: int = 9) -> dict:
    """Run MoGe inference on an image."""
    m = load_model()

    dtype = torch.float16 if settings.USE_FP16 else torch.float32
    image_tensor = torch.tensor(image, dtype=dtype, device='cuda').permute(2, 0, 1) / 255

    with torch.no_grad():
        output = m.infer(image_tensor, resolution_level=resolution_level, use_fp16=settings.USE_FP16)

    return {k: v.cpu().numpy() for k, v in output.items()}


def process_job(job_id: str, image: np.ndarray, params: dict):
    """Process a single job."""
    job_queue.update_job(job_id, status=JobStatus.PROCESSING)

    try:
        # Resize if needed
        height, width = image.shape[:2]
        max_size = params.get('max_size', settings.MAX_IMAGE_SIZE)
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        # Run inference
        resolution_level = params.get('resolution_level', settings.RESOLUTION_LEVEL)
        output = run_inference(image, resolution_level)

        points = output['points']
        depth = output['depth']
        mask = output['mask']
        normal = output.get('normal')
        intrinsics = output['intrinsics']

        # Clean mask (remove edges)
        if params.get('remove_edges', True):
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)
        else:
            mask_cleaned = mask

        # Create output directory for this job
        job_dir = settings.OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        outputs = {}
        requested = params.get('outputs', ['depth', 'normal', 'glb', 'ply'])

        # Save requested outputs
        if 'depth' in requested:
            depth_vis = colorize_depth(depth)
            depth_path = job_dir / 'depth.png'
            cv2.imwrite(str(depth_path), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            outputs['depth'] = str(depth_path)

            depth_exr_path = job_dir / 'depth.exr'
            cv2.imwrite(str(depth_exr_path), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            outputs['depth_exr'] = str(depth_exr_path)

        if 'normal' in requested and normal is not None:
            normal_vis = colorize_normal(normal)
            normal_path = job_dir / 'normal.png'
            cv2.imwrite(str(normal_path), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
            outputs['normal'] = str(normal_path)

        if 'glb' in requested or 'ply' in requested:
            # Build mesh
            if normal is not None:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            else:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None

            # Transform to OpenGL coordinates
            vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
            vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
            if vertex_normals is not None:
                vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)

            if 'glb' in requested:
                glb_path = job_dir / 'mesh.glb'
                # save_glb expects normal_texture (image), not vertex_normals
                normal_texture = (normal * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 255 if normal is not None else None
                normal_texture = normal_texture.astype(np.uint8) if normal_texture is not None else None
                save_glb(glb_path, vertices, faces, vertex_uvs, image, normal_texture)
                outputs['glb'] = str(glb_path)

            if 'ply' in requested:
                ply_path = job_dir / 'pointcloud.ply'
                save_ply(ply_path, vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)
                outputs['ply'] = str(ply_path)

        # Store points for measurement
        points_path = job_dir / 'points.npy'
        np.save(points_path, points)
        outputs['_points'] = str(points_path)

        # FOV info
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        outputs['fov'] = {
            'fov_x': float(np.rad2deg(fov_x)),
            'fov_y': float(np.rad2deg(fov_y))
        }

        job_queue.update_job(job_id, status=JobStatus.COMPLETED, outputs=outputs)

    except Exception as e:
        job_queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))


def worker_loop():
    """Background worker that processes jobs from the queue."""
    while True:
        job_id, image, params = job_queue.get_next()
        try:
            process_job(job_id, image, params)
        except Exception as e:
            job_queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))


# --- FastAPI App ---
app = FastAPI(title="MoGe 3D Generation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # Start worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    # Preload model
    load_model()


# --- Request/Response Models ---
class GenerateRequest(BaseModel):
    max_size: int = 800
    resolution_level: int = 9
    remove_edges: bool = True
    outputs: List[Literal['depth', 'normal', 'glb', 'ply']] = ['depth', 'normal', 'glb', 'ply']


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    outputs: Optional[dict] = None
    error: Optional[str] = None


class MeasureRequest(BaseModel):
    job_id: str
    point1: tuple[int, int]  # (x, y)
    point2: tuple[int, int]  # (x, y)


class MeasureResponse(BaseModel):
    distance_meters: float
    point1_depth: float
    point2_depth: float


# --- Endpoints ---
@app.post("/generate", response_model=JobResponse)
async def generate(
    file: UploadFile = File(...),
    max_size: int = 800,
    resolution_level: int = 9,
    remove_edges: bool = True,
    outputs: str = "depth,normal,glb,ply"
):
    """Submit an image for 3D generation. Returns a job ID for polling."""
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create job
    job_id = str(uuid.uuid4())
    job = Job(id=job_id)

    params = {
        'max_size': max_size,
        'resolution_level': resolution_level,
        'remove_edges': remove_edges,
        'outputs': [o.strip() for o in outputs.split(',')]
    }

    if not job_queue.add_job(job, image, params):
        raise HTTPException(status_code=503, detail="Queue is full, try again later")

    return JobResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        status=job.status,
        outputs=job.outputs if job.status == JobStatus.COMPLETED else None,
        error=job.error
    )


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated file."""
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    file_path = settings.OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@app.post("/measure", response_model=MeasureResponse)
async def measure_distance(request: MeasureRequest):
    """Measure distance between two points in a completed job."""
    job = job_queue.get_job(request.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    points_path = job.outputs.get('_points')
    if not points_path or not Path(points_path).exists():
        raise HTTPException(status_code=400, detail="Points data not available")

    points = np.load(points_path)
    x1, y1 = request.point1
    x2, y2 = request.point2

    # Bounds check
    h, w = points.shape[:2]
    if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
        raise HTTPException(status_code=400, detail="Points out of bounds")

    p1 = points[y1, x1]
    p2 = points[y2, x2]
    distance = float(np.linalg.norm(p1 - p2))

    return MeasureResponse(
        distance_meters=distance,
        point1_depth=float(p1[2]),
        point2_depth=float(p2[2])
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
