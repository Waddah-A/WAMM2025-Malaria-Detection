"""
WAMM2025 - Malaria Detection System
Dataset 1: NIH Malaria Dataset

27,558 images - The largest and most important dataset
Source: https://data.lhncbc.nlm.nih.gov/public/Malaria/
"""

import os
import zipfile

print("=" * 80)
print("DATASET 1/6: NIH DATASET")
print("27,558 images (350 MB)")
print("=" * 80)

nih_path = "/content/cell_images"

if os.path.exists(nih_path):
    print("NIH dataset already downloaded!")

    parasitized = len([f for f in os.listdir(os.path.join(nih_path, "Parasitized"))
                       if f.endswith(('.png', '.jpg'))])
    uninfected = len([f for f in os.listdir(os.path.join(nih_path, "Uninfected"))
                      if f.endswith(('.png', '.jpg'))])

    print(f"  Parasitized: {parasitized:,}")
    print(f"  Uninfected:  {uninfected:,}")
    print(f"  Total:       {parasitized + uninfected:,}")
else:
    print("Downloading NIH dataset...")
    print("  Size: 350 MB")
    print("  Time: ~1-2 minutes")

    os.system('wget -q --show-progress https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip')

    print("Extracting...")
    with zipfile.ZipFile('cell_images.zip', 'r') as zip_ref:
        zip_ref.extractall('/content')

    os.remove('cell_images.zip')

    if os.path.exists(nih_path):
        parasitized = len(os.listdir(os.path.join(nih_path, "Parasitized")))
        uninfected = len(os.listdir(os.path.join(nih_path, "Uninfected")))

        print(f"NIH DATASET READY!")
        print(f"  Parasitized: {parasitized:,}")
        print(f"  Uninfected:  {uninfected:,}")
        print(f"  Total:       {parasitized + uninfected:,}")
    else:
        print("Download failed - please run this cell again")

print()
print("=" * 80)
print("Dataset 1/6 complete!")
print("=" * 80)
