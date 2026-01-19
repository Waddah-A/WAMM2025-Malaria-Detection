"""
WAMM2025 - Malaria Detection System
Fix Corrupted Images

Scans and removes corrupted image files before training.
Run this BEFORE training (code_11_train.py).
"""

import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("FIXING CORRUPTED IMAGES")
print("Scanning dataset for corrupted files...")
print("=" * 80)

MERGED_PATH = "/content/combined_multi_species"


def check_and_remove_corrupted_images(directory):
    """Check all images in directory and remove corrupted ones."""
    corrupted = []
    total_checked = 0

    print(f"\nScanning: {directory}")

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF']:
        image_files.extend(Path(directory).rglob(ext))

    print(f"  Found {len(image_files)} image files")
    print(f"  Checking each one...")

    for img_path in tqdm(image_files, desc="  Checking"):
        total_checked += 1
        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                img.load()

        except Exception as e:
            corrupted.append({
                'path': str(img_path),
                'error': str(e)
            })

    return corrupted, total_checked


# Check Parasitized folder
print()
print("=" * 80)
print("CHECKING PARASITIZED IMAGES")
print("=" * 80)

parasitized_path = os.path.join(MERGED_PATH, "Parasitized")
corrupted_parasitized, total_parasitized = check_and_remove_corrupted_images(parasitized_path)

if corrupted_parasitized:
    print(f"\nFound {len(corrupted_parasitized)} corrupted images")
else:
    print(f"\nAll {total_parasitized} images are valid!")

# Check Uninfected folder
print()
print("=" * 80)
print("CHECKING UNINFECTED IMAGES")
print("=" * 80)

uninfected_path = os.path.join(MERGED_PATH, "Uninfected")
corrupted_uninfected, total_uninfected = check_and_remove_corrupted_images(uninfected_path)

if corrupted_uninfected:
    print(f"\nFound {len(corrupted_uninfected)} corrupted images")
else:
    print(f"\nAll {total_uninfected} images are valid!")

# Total corrupted
total_corrupted = len(corrupted_parasitized) + len(corrupted_uninfected)
all_corrupted = corrupted_parasitized + corrupted_uninfected

# Summary
print()
print("=" * 80)
print("SCAN RESULTS")
print("=" * 80)

print(f"\nTotal images checked: {total_parasitized + total_uninfected:,}")
print(f"  Parasitized: {total_parasitized:,}")
print(f"  Uninfected:  {total_uninfected:,}")

if total_corrupted > 0:
    print(f"\nCorrupted images found: {total_corrupted}")
    print(f"\nCorrupted file details:")
    for i, item in enumerate(all_corrupted[:10], 1):
        print(f"  {i}. {Path(item['path']).name}")
        print(f"     Error: {item['error'][:60]}...")

    if len(all_corrupted) > 10:
        print(f"  ... and {len(all_corrupted) - 10} more")

    print("\nRemoving corrupted files...")
    removed_count = 0

    for item in tqdm(all_corrupted, desc="  Removing"):
        try:
            os.remove(item['path'])
            removed_count += 1
        except Exception as e:
            print(f"    Could not remove {Path(item['path']).name}: {e}")

    print(f"\nRemoved {removed_count}/{total_corrupted} corrupted files")

    final_parasitized = len([f for f in os.listdir(parasitized_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    final_uninfected = len([f for f in os.listdir(uninfected_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    print(f"\nCLEANED DATASET:")
    print(f"  Parasitized: {final_parasitized:,} images")
    print(f"  Uninfected:  {final_uninfected:,} images")
    print(f"  Total:       {final_parasitized + final_uninfected:,} images")

    print()
    print("=" * 80)
    print("DATASET CLEANED!")
    print("=" * 80)
    print("\nNow run code_11_train.py to train Stage 1")

else:
    print(f"\nNO CORRUPTED IMAGES FOUND!")
    print(f"All {total_parasitized + total_uninfected:,} images are valid")
    print(f"\nDataset is clean and ready for training!")
    print(f"\nTip: The error might be from a different issue.")
    print(f"Try running code_11_train.py again.")

print()
print("=" * 80)

# Additional checks
print()
print("ADDITIONAL DIAGNOSTICS:")
print("=" * 80)

parasitized_count = len(os.listdir(parasitized_path))
uninfected_count = len(os.listdir(uninfected_path))

if parasitized_count == 0:
    print("[FAIL] Parasitized folder is EMPTY!")
elif uninfected_count == 0:
    print("[FAIL] Uninfected folder is EMPTY!")
else:
    print(f"[OK] Both folders have files")
    print(f"  Parasitized: {parasitized_count:,} files")
    print(f"  Uninfected:  {uninfected_count:,} files")

print()
print("Checking for suspiciously small files...")
small_files = []

for folder in [parasitized_path, uninfected_path]:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            if size < 1000:
                small_files.append((file, size))

if small_files:
    print(f"Found {len(small_files)} suspiciously small files (< 1KB):")
    for filename, size in small_files[:5]:
        print(f"  - {filename}: {size} bytes")

    print()
    print("Removing small files...")
    for filename, size in small_files:
        for folder in [parasitized_path, uninfected_path]:
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"  Removed {filename}")
                except:
                    pass
else:
    print("[OK] No suspiciously small files found")

print()
print("=" * 80)
print("DATASET IS NOW READY!")
print("=" * 80)
print("\nNext step: Run code_11_train.py (Train Stage 1)")
