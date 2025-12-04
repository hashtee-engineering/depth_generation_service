#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Install/upgrade pip
echo "Upgrading pip..."
$PIP install --upgrade pip

# Install package in editable mode
echo "Installing dependencies..."
$PIP install -e .

# Start server
echo "Starting server..."
exec $PYTHON -m uvicorn server:app --host 0.0.0.0 --port 8000
