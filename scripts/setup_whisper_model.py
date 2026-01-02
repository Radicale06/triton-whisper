#!/usr/bin/env python3
"""
Setup Whisper v3 Turbo for NVIDIA Triton Inference Server
=========================================================

This script prepares OpenAI's Whisper v3 Turbo model to run on Triton
using the Python backend for maximum flexibility.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_whisper_for_triton():
    """Setup Whisper v3 Turbo model files for Triton."""
    
    model_name = "openai/whisper-large-v3-turbo"
    model_dir = Path("/workspace/model_repository/whisper_v3_turbo/1")
    
    logger.info(f"Setting up {model_name} for Triton...")
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a marker file to indicate model should be downloaded on first run
    marker_file = model_dir / "model_info.txt"
    with open(marker_file, "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write("backend: python\n")
        f.write("description: Whisper v3 Turbo for speech recognition\n")
    
    logger.info(f"Model setup complete. Marker file created at {marker_file}")
    logger.info("Model will be downloaded on first inference request to save space")


if __name__ == "__main__":
    setup_whisper_for_triton()