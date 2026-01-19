"""
WAMM2025 - Malaria Detection System
Initial Setup - GPU Check & Package Installation

Run this first in your Colab notebook.
"""

import tensorflow as tf
import os

print("=" * 80)
print("WAMM2025: Complete 2-Stage Malaria Detection System")
print("Training on ALL 6 DATASETS (~38,000 images)")
print("=" * 80)
print()
print("Dataset Overview:")
print("  Dataset 1: NIH (27,558 images)")
print("  Dataset 2: MP-IDB (229 images - 4 species)")
print("  Dataset 3: Broad Institute (1,364 images)")
print("  Dataset 4: Tek (655 images + bounding boxes)")
print("  Dataset 5: Lacuna - Ghana + Uganda (~3,314 images)")
print("  Dataset 6: Tanzania NM-AIST (3,544 images)")
print()

# Check GPU
print("=" * 80)
print("GPU CHECK")
print("=" * 80)

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU Available: {gpu_devices[0].name}")
    print("Training will be FAST (~30 min Stage 1, ~45 min Stage 2)")
else:
    print("No GPU detected!")
    print("Action: Runtime -> Change runtime type -> T4 GPU -> Save")
    print("Then restart this cell")

# Install packages
print()
print("=" * 80)
print("INSTALLING REQUIRED PACKAGES")
print("=" * 80)

print("Installing Ultralytics (YOLO)...")
os.system('pip install -q ultralytics')

print("Installing Kaggle API...")
os.system('pip install -q kaggle')

print("Installing other dependencies...")
os.system('pip install -q gdown Pillow')

print()
print("All packages installed!")
print("=" * 80)
print("Ready to download datasets!")
print("=" * 80)
