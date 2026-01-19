"""
WAMM2025 - Malaria Detection System
Dataset 2: MP-IDB Multi-Species Database

229 images covering all 4 malaria species
Source: https://github.com/andrealoddo/MP-IDB-...
"""

import os
import shutil

print("=" * 80)
print("DATASET 2/6: MP-IDB DATASET")
print("229 images - All 4 species")
print("=" * 80)

mpidb_path = "/content/mpidb_extracted"

if os.path.exists(mpidb_path):
    print("MP-IDB dataset already downloaded!")

    img_count = sum([len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
                     for _, _, files in os.walk(mpidb_path)])
    print(f"  Images: {img_count:,}")
else:
    print("Cloning MP-IDB repository from GitHub...")
    print("  Size: ~50 MB")
    print("  Time: ~30 seconds")

    os.system(f'git clone https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.git {mpidb_path}')

    if os.path.exists(mpidb_path):
        img_count = sum([len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
                         for _, _, files in os.walk(mpidb_path)])

        print(f"MP-IDB DATASET READY!")
        print(f"  Images: {img_count:,}")
        print(f"  Species coverage:")
        print(f"    - P. falciparum")
        print(f"    - P. vivax")
        print(f"    - P. ovale")
        print(f"    - P. malariae")
    else:
        print("Clone failed - please run this cell again")

print()
print("=" * 80)
print("Dataset 2/6 complete!")
print("=" * 80)
