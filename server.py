import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import uuid
import tempfile
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Literal
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
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
    DATA_DIR: Path = Path(tempfile.gettempdir()) / "moge_data"
    DB_PATH: Path = DATA_DIR / "jobs.db"
    FILE_RETENTION_SECONDS: int = 86400  # 24 hours
    MAX_QUEUE_SIZE: int = 100


settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)


# --- SQLite Database ---
def get_db_connection():
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize SQLite database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            created_at REAL NOT NULL,
            completed_at REAL,
            image_width INTEGER,
            image_height INTEGER,
            fov_x REAL,
            fov_y REAL,
            params TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            output_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_outputs_job ON job_outputs(job_id)
    ''')

    conn.commit()
    conn.close()


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DatabaseManager:
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()

    def get_connection(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def create_job(self, job_id: str, params: dict) -> bool:
        conn = self.get_connection()
        try:
            conn.execute(
                'INSERT INTO jobs (id, status, created_at, params) VALUES (?, ?, ?, ?)',
                (job_id, JobStatus.PENDING.value, time.time(), json.dumps(params))
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error creating job: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[dict]:
        conn = self.get_connection()
        row = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def update_job(self, job_id: str, **kwargs):
        conn = self.get_connection()
        set_clause = ', '.join(f'{k} = ?' for k in kwargs.keys())
        values = list(kwargs.values()) + [job_id]
        conn.execute(f'UPDATE jobs SET {set_clause} WHERE id = ?', values)
        conn.commit()

    def add_output(self, job_id: str, output_type: str, file_path: str):
        conn = self.get_connection()
        conn.execute(
            'INSERT INTO job_outputs (job_id, output_type, file_path, created_at) VALUES (?, ?, ?, ?)',
            (job_id, output_type, file_path, time.time())
        )
        conn.commit()

    def get_outputs(self, job_id: str) -> dict:
        conn = self.get_connection()
        rows = conn.execute(
            'SELECT output_type, file_path FROM job_outputs WHERE job_id = ?',
            (job_id,)
        ).fetchall()
        return {row['output_type']: row['file_path'] for row in rows}

    def get_output_path(self, job_id: str, output_type: str) -> Optional[str]:
        conn = self.get_connection()
        row = conn.execute(
            'SELECT file_path FROM job_outputs WHERE job_id = ? AND output_type = ?',
            (job_id, output_type)
        ).fetchone()
        return row['file_path'] if row else None

    def cleanup_old_jobs(self, max_age_seconds: int):
        """Remove jobs older than max_age_seconds."""
        conn = self.get_connection()
        cutoff = time.time() - max_age_seconds

        # Get old jobs
        old_jobs = conn.execute(
            'SELECT id FROM jobs WHERE created_at < ?', (cutoff,)
        ).fetchall()

        for job in old_jobs:
            job_id = job['id']
            # Delete job directory
            job_dir = settings.DATA_DIR / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)

        # Delete from database
        conn.execute('DELETE FROM job_outputs WHERE job_id IN (SELECT id FROM jobs WHERE created_at < ?)', (cutoff,))
        conn.execute('DELETE FROM jobs WHERE created_at < ?', (cutoff,))
        conn.commit()

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[dict]:
        conn = self.get_connection()
        if status:
            rows = conn.execute(
                'SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?',
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


db = DatabaseManager()


# --- Job Queue (in-memory for processing) ---
class JobQueue:
    def __init__(self, max_size: int = 100):
        self.queue: deque = deque()
        self.lock = threading.Lock()
        self.max_size = max_size
        self.condition = threading.Condition(self.lock)

    def add_job(self, job_id: str, image: np.ndarray, params: dict) -> bool:
        with self.lock:
            if len(self.queue) >= self.max_size:
                return False
            self.queue.append((job_id, image, params))
            self.condition.notify()
            return True

    def get_next(self) -> Optional[tuple]:
        with self.condition:
            while len(self.queue) == 0:
                self.condition.wait()
            return self.queue.popleft()


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
    db.update_job(job_id, status=JobStatus.PROCESSING.value)

    try:
        # Create output directory for this job
        job_dir = settings.DATA_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = job_dir / 'original.jpg'
        cv2.imwrite(str(original_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        db.add_output(job_id, 'original', str(original_path))

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

        requested = params.get('outputs', ['depth', 'normal', 'glb', 'ply'])

        # Save requested outputs
        if 'depth' in requested:
            depth_vis = colorize_depth(depth)
            depth_path = job_dir / 'depth.png'
            cv2.imwrite(str(depth_path), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            db.add_output(job_id, 'depth', str(depth_path))

            depth_exr_path = job_dir / 'depth.exr'
            cv2.imwrite(str(depth_exr_path), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            db.add_output(job_id, 'depth_exr', str(depth_exr_path))

        if 'normal' in requested and normal is not None:
            normal_vis = colorize_normal(normal)
            normal_path = job_dir / 'normal.png'
            cv2.imwrite(str(normal_path), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
            db.add_output(job_id, 'normal', str(normal_path))

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
                db.add_output(job_id, 'glb', str(glb_path))

            if 'ply' in requested:
                ply_path = job_dir / 'pointcloud.ply'
                save_ply(ply_path, vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)
                db.add_output(job_id, 'ply', str(ply_path))

        # Save points as numpy file (for server-side measurement)
        points_npy_path = job_dir / 'points.npy'
        np.save(points_npy_path, points)
        db.add_output(job_id, 'points_npy', str(points_npy_path))

        # Save points as JSON (for mobile app 3D viewer)
        # Flatten points array and convert to list for JSON serialization
        # points shape is (H, W, 3), we need to flatten to list of [x, y, z]
        h, w = points.shape[:2]
        points_list = points.reshape(-1, 3).tolist()
        points_json = {
            'width': w,
            'height': h,
            'points': points_list
        }
        points_json_path = job_dir / 'points.json'
        with open(points_json_path, 'w') as f:
            json.dump(points_json, f)
        db.add_output(job_id, 'points_json', str(points_json_path))

        # FOV info
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        fov_x_deg = float(np.rad2deg(fov_x))
        fov_y_deg = float(np.rad2deg(fov_y))

        db.update_job(
            job_id,
            status=JobStatus.COMPLETED.value,
            completed_at=time.time(),
            image_width=w,
            image_height=h,
            fov_x=fov_x_deg,
            fov_y=fov_y_deg
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.update_job(job_id, status=JobStatus.FAILED.value, error=str(e))


def worker_loop():
    """Background worker that processes jobs from the queue."""
    while True:
        job_id, image, params = job_queue.get_next()
        try:
            process_job(job_id, image, params)
        except Exception as e:
            db.update_job(job_id, status=JobStatus.FAILED.value, error=str(e))


def cleanup_loop():
    """Background worker that cleans up old jobs."""
    while True:
        time.sleep(3600)  # Run every hour
        try:
            db.cleanup_old_jobs(settings.FILE_RETENTION_SECONDS)
            print(f"Cleaned up jobs older than {settings.FILE_RETENTION_SECONDS} seconds")
        except Exception as e:
            print(f"Cleanup error: {e}")


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
    # Initialize database
    init_database()
    # Start worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
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
    status: str
    outputs: Optional[dict] = None
    error: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    fov_x: Optional[float] = None
    fov_y: Optional[float] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None


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

    params = {
        'max_size': max_size,
        'resolution_level': resolution_level,
        'remove_edges': remove_edges,
        'outputs': [o.strip() for o in outputs.split(',')]
    }

    if not db.create_job(job_id, params):
        raise HTTPException(status_code=500, detail="Failed to create job")

    if not job_queue.add_job(job_id, image, params):
        db.update_job(job_id, status=JobStatus.FAILED.value, error="Queue is full")
        raise HTTPException(status_code=503, detail="Queue is full, try again later")

    return JobResponse(job_id=job_id, status=JobStatus.PENDING.value)


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    outputs = None
    if job['status'] == JobStatus.COMPLETED.value:
        outputs = db.get_outputs(job_id)
        # Convert file paths to download URLs
        outputs = {k: f"/download/{job_id}/{Path(v).name}" for k, v in outputs.items() if not k.startswith('_')}

    return JobResponse(
        job_id=job['id'],
        status=job['status'],
        outputs=outputs,
        error=job['error'],
        image_width=job['image_width'],
        image_height=job['image_height'],
        fov_x=job['fov_x'],
        fov_y=job['fov_y'],
        created_at=job['created_at'],
        completed_at=job['completed_at']
    )


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List all jobs, optionally filtered by status."""
    jobs = db.list_jobs(status=status, limit=limit)
    return {"jobs": jobs}


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated file."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    file_path = settings.DATA_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Set appropriate content type
    content_type = "application/octet-stream"
    if filename.endswith('.png'):
        content_type = "image/png"
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        content_type = "image/jpeg"
    elif filename.endswith('.json'):
        content_type = "application/json"
    elif filename.endswith('.glb'):
        content_type = "model/gltf-binary"
    elif filename.endswith('.ply'):
        content_type = "application/x-ply"
    elif filename.endswith('.exr'):
        content_type = "image/x-exr"

    return FileResponse(file_path, filename=filename, media_type=content_type)


@app.post("/measure", response_model=MeasureResponse)
async def measure_distance(request: MeasureRequest):
    """Measure distance between two points in a completed job."""
    job = db.get_job(request.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    points_path = db.get_output_path(request.job_id, 'points_npy')
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


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    job_dir = settings.DATA_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)

    # Delete from database
    conn = db.get_connection()
    conn.execute('DELETE FROM job_outputs WHERE job_id = ?', (job_id,))
    conn.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    conn.commit()

    return {"status": "deleted", "job_id": job_id}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "database": str(settings.DB_PATH),
        "data_dir": str(settings.DATA_DIR)
    }


@app.get("/stats")
async def stats():
    """Get service statistics."""
    conn = db.get_connection()

    total = conn.execute('SELECT COUNT(*) as count FROM jobs').fetchone()['count']
    pending = conn.execute('SELECT COUNT(*) as count FROM jobs WHERE status = ?', (JobStatus.PENDING.value,)).fetchone()['count']
    processing = conn.execute('SELECT COUNT(*) as count FROM jobs WHERE status = ?', (JobStatus.PROCESSING.value,)).fetchone()['count']
    completed = conn.execute('SELECT COUNT(*) as count FROM jobs WHERE status = ?', (JobStatus.COMPLETED.value,)).fetchone()['count']
    failed = conn.execute('SELECT COUNT(*) as count FROM jobs WHERE status = ?', (JobStatus.FAILED.value,)).fetchone()['count']

    return {
        "total_jobs": total,
        "pending": pending,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "queue_size": len(job_queue.queue)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
