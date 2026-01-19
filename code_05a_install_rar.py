"""
WAMM2025 - Malaria Detection System
Install RAR Dependencies

Required for extracting Lacuna dataset RAR files
Run this ONCE before Dataset 5 code
"""

import os

print("Installing RAR extraction tools...")
print("=" * 80)

print("1. Installing unrar (system utility)...")
os.system('apt-get install -y unrar')

print("2. Installing rarfile (Python module)...")
os.system('pip install -q rarfile')

print()
print("RAR extraction tools installed!")
print("=" * 80)
print("Now run your Dataset 5 code!")
