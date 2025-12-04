# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI-based 3D geometry generation service using MoGe. Accepts images and returns depth maps, normal maps, 3D models, and metric measurements. Uses SQLite for job persistence.

## Commands

```bash
# Setup and start server
./start.sh

# Or manually:
pip install -e .
uvicorn server:app --host 0.0.0.0 --port 8000

# With Gradio demo (optional):
pip install -e ".[gradio]"
python gradio_demo.py
```

## API Endpoints

- `POST /generate` - Submit image for 3D generation, returns job_id
- `GET /job/{job_id}` - Poll job status (includes image_width, image_height, fov_x, fov_y)
- `GET /jobs` - List all jobs (optional status filter)
- `GET /download/{job_id}/{filename}` - Download generated files
- `POST /measure` - Measure distance between two pixel coordinates
- `DELETE /job/{job_id}` - Delete a job and its files
- `GET /health` - Health check
- `GET /stats` - Service statistics

## Generated Files

Each completed job produces:
- `original.jpg` - Original uploaded image
- `normal.png` - Colorized normal map
- `depth.png` - Colorized depth map
- `depth.exr` - Raw depth values (OpenEXR)
- `mesh.glb` - 3D mesh (GLB format)
- `pointcloud.ply` - Point cloud (PLY format)
- `points.npy` - Raw 3D points (NumPy, server-side measurement)
- `points.json` - 3D points for mobile app (JSON with width, height, points array)

## Database

SQLite database at `{DATA_DIR}/jobs.db`:
- `jobs` table: id, status, error, created_at, completed_at, image_width, image_height, fov_x, fov_y, params
- `job_outputs` table: job_id, output_type, file_path, created_at

## Key Files

- `server.py` - FastAPI server with SQLite, queue management, cleanup
- `test.html` - Browser-based test client
- `moge/model/v2.py` - MoGe model with normal estimation and metric scale
- `moge/utils/io.py` - `save_glb()`, `save_ply()` for 3D export
- `moge/utils/vis.py` - `colorize_depth()`, `colorize_normal()`
- `gradio_demo.py` - Reference Gradio implementation

## MoGe Model

```python
from moge.model import import_model_class_by_version
model = import_model_class_by_version('v2').from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
```

Inference output:
- `points`: 3D point map (H, W, 3) - metric scale in meters
- `depth`: Depth map (H, W)
- `normal`: Surface normals (H, W, 3)
- `mask`: Valid pixel mask (H, W)
- `intrinsics`: Camera intrinsics (3, 3)

## Measurement

Distance between two image points:
```python
distance = np.linalg.norm(points[y1, x1] - points[y2, x2])  # meters
```

## points.json Format

```json
{
  "width": 800,
  "height": 600,
  "points": [[x1, y1, z1], [x2, y2, z2], ...]
}
```

Points are flattened row-major: index = y * width + x
