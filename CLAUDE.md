# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI-based 3D geometry generation service using MoGe. Accepts images and returns depth maps, normal maps, 3D models, and metric measurements.

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

## MoGe Model

```python
from moge.model import import_model_class_by_version
model = import_model_class_by_version('v2').from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
```

Inference output:
- `points`: 3D point map (H, W, 3)
- `depth`: Depth map (H, W)
- `normal`: Surface normals (H, W, 3)
- `mask`: Valid pixel mask (H, W)
- `intrinsics`: Camera intrinsics (3, 3)

## Key Files

- `moge/model/v2.py`: Model with normal estimation and metric scale
- `moge/utils/io.py`: `save_glb()`, `save_ply()` for 3D export
- `moge/utils/vis.py`: `colorize_depth()`, `colorize_normal()`
- `gradio_demo.py`: Reference Gradio implementation
