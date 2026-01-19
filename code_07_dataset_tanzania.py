"""
WAMM2025 - Malaria Detection System
Dataset 6: NM-AIST Tanzania Malaria Dataset

3,544 high-resolution images (3840x2160 pixels)
AUTO-DOWNLOAD FROM GOOGLE DRIVE

Source: Harvard Dataverse - Nelson Mandela African Institution
DOI: 10.7910/DVN/O2WVWA
"""

import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("DATASET 6/6: NM-AIST (TANZANIA)")
print("3,544 HIGH-RES images - AUTO DOWNLOAD")
print("DOI: 10.7910/DVN/O2WVWA")
print("=" * 80)

tanzania_path = "/content/tanzania_extracted"

if os.path.exists(tanzania_path):
    categories = {
        'THICK_INFECTED': 0,
        'THICK_UNINFECTED': 0,
        'THIN_INFECTED': 0,
        'THIN_UNINFECTED': 0
    }
    
    base_path = os.path.join(tanzania_path, "tanzania_dataset")
    if os.path.exists(base_path):
        for cat in categories.keys():
            for variant in [cat, cat.replace('_', ' '), cat.title().replace('_', '_')]:
                cat_path = os.path.join(base_path, variant)
                if os.path.exists(cat_path):
                    categories[cat] = len([f for f in os.listdir(cat_path) 
                                          if f.lower().endswith(('.jpg', '.png', '.jpeg')) 
                                          and not f.startswith('._')])
                    break
    
    total = sum(categories.values())
    
    if total > 0:
        print("Tanzania NM-AIST dataset already downloaded!")
        print(f"  Total images: {total:,}")
        for cat, count in categories.items():
            if count > 0:
                print(f"  - {cat}: {count:,}")
    else:
        print("Dataset folder exists but appears empty - will re-process")

if not os.path.exists(tanzania_path) or sum(categories.values()) == 0:
    print("AUTO-DOWNLOADING FROM GOOGLE DRIVE...")
    print("=" * 80)
    print()
    print("Dataset Information:")
    print("  - Resolution: 3840x2160 pixels (4K)")
    print("  - Total: 3,544 images")
    print("  - Microscope: 40X-2500X compound with Sony IMX334")
    print("  - Staining: Giemsa reagent")
    print("  - Source: 5 health centers in Tanga region, Tanzania")
    print()
    print("=" * 80)

    try:
        print("Installing gdown...")
        os.system('pip install -q gdown')
        import gdown
        
        file_id = "14e8Muiom2LZpDmhixyF2_zKVcC12hs-C"
        output_file = "/content/tanzania_dataset.zip"
        
        if not os.path.exists(output_file):
            print(f"Downloading tanzania_dataset.zip from Google Drive...")
            print(f"  File ID: {file_id}")
            print(f"  This may take several minutes...")
            
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_file, quiet=False)
            
            if not os.path.exists(output_file):
                raise Exception("Download failed - file not found")
            
            file_size = os.path.getsize(output_file) / (1024**2)
            print(f"Downloaded successfully! Size: {file_size:.1f} MB")
        else:
            print(f"ZIP file already downloaded, skipping download")
        
        os.makedirs(tanzania_path, exist_ok=True)
        
        print(f"Extracting ZIP file...")

        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(tanzania_path)

        os.remove(output_file)
        print("ZIP extracted")

        base_path = os.path.join(tanzania_path, "tanzania_dataset")
        
        if not os.path.exists(base_path):
            base_path = tanzania_path
        
        print(f"Processing dataset structure...")
        print(f"  Base path: {base_path}")
        
        categories = {
            'Thick_Infected': 0,
            'Thick_Uninfected': 0,
            'Thin_Infected': 0,
            'Thin_Uninfected': 0
        }
        
        for cat in categories.keys():
            cat_path = os.path.join(base_path, cat)
            if os.path.exists(cat_path):
                categories[cat] = len([f for f in os.listdir(cat_path) 
                                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                                      and not f.startswith('._')])
            else:
                for root, dirs, _ in os.walk(base_path):
                    if cat.lower() in [d.lower() for d in dirs]:
                        actual_name = [d for d in dirs if d.lower() == cat.lower()][0]
                        cat_path = os.path.join(root, actual_name)
                        categories[cat] = len([f for f in os.listdir(cat_path) 
                                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                                              and not f.startswith('._')])
                        break

        total_images = sum(categories.values())

        print(f"EXTRACTION COMPLETE!")
        print(f"  Total images: {total_images:,}")
        for cat, count in categories.items():
            if count > 0:
                print(f"  - {cat}: {count:,}")

        if total_images >= 3000:
            print(f"TANZANIA DATASET READY!")
            print(f"  - High-resolution 4K images")
            print(f"  - Balanced infected/uninfected data")
            print(f"  - Both thick & thin smears")
            print(f"  Source: Tanga region health centers, Tanzania")

        elif total_images >= 2000:
            print(f"TANZANIA DATASET LOADED!")
            print(f"  Found {total_images:,} images")
            print(f"  Expected ~3,544 images")

        else:
            print(f"Expected ~3,544 images, found {total_images:,}")
            print(f"Dataset may need verification")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("  1. Check Google Drive link is publicly accessible")
        print("  2. Verify file ID is correct")
        print("  3. Check internet connection")
        print("  4. Try re-running this cell")

print()
print("=" * 80)
print("Dataset 6/6 complete!")
print("=" * 80)

print()
print("DATASET DETAILS:")
print("  - Source: Nelson Mandela African Institution, Tanzania")
print("  - Quality: 4K resolution (3840x2160 pixels)")
print("  - Coverage: Complete coverage with 4 categories")
print("  - Species: Primarily P. falciparum (Tanzania)")
print("  - Use: Binary classification with high-resolution detail")

print()
print("CLINICAL SIGNIFICANCE:")
print("  - Highest resolution in your dataset (4K)")
print("  - Professional microscopy equipment")
print("  - Balanced infected/uninfected samples")
print("  - Represents East African malaria patterns")
print("  - Excellent for training high-accuracy models")
