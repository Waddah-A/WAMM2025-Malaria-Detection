"""
WAMM2025 - Malaria Detection System
Dataset 5: Lacuna Malaria Dataset (Ghana + Uganda)

~8,000 images total from 2 countries
AUTO-DOWNLOAD FROM GOOGLE DRIVE (REORGANIZED VERSION)

Source: Harvard Dataverse - Lacuna Malaria Project
DOI: 10.7910/DVN/VEADSE
"""

import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd

print("=" * 80)
print("DATASET 5/6: LACUNA (GHANA + UGANDA)")
print("~8,000 images - REORGANIZED VERSION")
print("DOI: 10.7910/DVN/VEADSE")
print("=" * 80)

lacuna_path = "/content/lacuna_extracted"
lacuna_yolo_path = "/content/lacuna_yolo_annotations"

if os.path.exists(lacuna_path):
    print("Removing old data...")
    shutil.rmtree(lacuna_path, ignore_errors=True)
if os.path.exists(lacuna_yolo_path):
    shutil.rmtree(lacuna_yolo_path, ignore_errors=True)

print("AUTO-DOWNLOADING FROM GOOGLE DRIVE...")
print("=" * 80)
print()
print("Dataset Information:")
print("  - Source: Lacuna Malaria Project (Ghana + Uganda)")
print("  - Countries: Ghana (MinoHealth) + Uganda (Makerere AI Lab)")
print("  - Structure: Thick_Ghana, Thin_Ghana, Thin_Uganda")
print("  - Includes: YOLO bounding box annotations")
print()
print("=" * 80)

try:
    print("Installing gdown...")
    os.system('pip install -q gdown')
    import gdown
    
    file_id = "1wPeK7xBcWSDB8VcEnlJUkjiWpXKnc77R"
    output_file = "/content/lacuna_dataset.zip"
    
    if not os.path.exists(output_file):
        print(f"Downloading lacuna_dataset.zip from Google Drive...")
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
    
    temp_extract = "/content/lacuna_temp"
    os.makedirs(temp_extract, exist_ok=True)
    os.makedirs(lacuna_path, exist_ok=True)
    os.makedirs(lacuna_yolo_path, exist_ok=True)
    
    print(f"Extracting ZIP file...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
    
    os.remove(output_file)
    print("ZIP extracted")
    
    base_path = os.path.join(temp_extract, "lacuna_dataset")
    
    if not os.path.exists(base_path):
        base_path = temp_extract
    
    print(f"Processing reorganized dataset structure...")
    print(f"  Base path: {base_path}")
    
    ghana_thick_count = 0
    ghana_thin_count = 0
    uganda_thin_count = 0
    total_annotations = 0
    
    # PROCESS THICK_GHANA
    thick_ghana_path = os.path.join(base_path, "Thick_Ghana")
    if os.path.exists(thick_ghana_path):
        print(f"Processing Thick_Ghana...")
        
        for img_file in tqdm(os.listdir(thick_ghana_path), desc="   Thick_Ghana"):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(thick_ghana_path, img_file)
                clean_name = img_file.replace(' ', '_').replace('(', '').replace(')', '')
                new_name = f"lacuna_ghana_thick_{clean_name}"
                dst = os.path.join(lacuna_path, new_name)
                
                try:
                    shutil.copy2(src, dst)
                    ghana_thick_count += 1
                except:
                    pass
    else:
        print(f"Warning: Thick_Ghana not found at {thick_ghana_path}")
    
    # PROCESS THIN_GHANA
    thin_ghana_path = os.path.join(base_path, "Thin_Ghana")
    if os.path.exists(thin_ghana_path):
        print(f"Processing Thin_Ghana...")
        
        images_path = os.path.join(thin_ghana_path, "images")
        labels_path = os.path.join(thin_ghana_path, "labels_yolo")
        
        if os.path.exists(images_path):
            for img_file in tqdm(os.listdir(images_path), desc="   Thin_Ghana"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(images_path, img_file)
                    clean_name = img_file.replace(' ', '_').replace('(', '').replace(')', '')
                    new_name = f"lacuna_ghana_thin_{clean_name}"
                    dst = os.path.join(lacuna_path, new_name)
                    
                    try:
                        shutil.copy2(src, dst)
                        ghana_thin_count += 1
                        
                        if os.path.exists(labels_path):
                            img_basename = Path(img_file).stem
                            label_file = img_basename + '.txt'
                            label_src = os.path.join(labels_path, label_file)
                            
                            if os.path.exists(label_src):
                                label_dst = os.path.join(lacuna_yolo_path, 
                                                        Path(new_name).stem + '.txt')
                                try:
                                    shutil.copy2(label_src, label_dst)
                                    total_annotations += 1
                                except:
                                    pass
                    except:
                        pass
        else:
            print(f"Warning: Thin_Ghana/images not found at {images_path}")
    else:
        print(f"Warning: Thin_Ghana not found at {thin_ghana_path}")
    
    # PROCESS THIN_UGANDA
    thin_uganda_path = os.path.join(base_path, "Thin_Uganda")
    if os.path.exists(thin_uganda_path):
        print(f"Processing Thin_Uganda...")
        
        images_path = os.path.join(thin_uganda_path, "images")
        labels_path = os.path.join(thin_uganda_path, "Labels-YOLO")
        
        if os.path.exists(images_path):
            for img_file in tqdm(os.listdir(images_path), desc="   Thin_Uganda"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(images_path, img_file)
                    clean_name = img_file.replace(' ', '_').replace('(', '').replace(')', '')
                    new_name = f"lacuna_uganda_thin_{clean_name}"
                    dst = os.path.join(lacuna_path, new_name)
                    
                    try:
                        shutil.copy2(src, dst)
                        uganda_thin_count += 1
                        
                        if os.path.exists(labels_path):
                            img_basename = Path(img_file).stem
                            label_file = img_basename + '.txt'
                            label_src = os.path.join(labels_path, label_file)
                            
                            if os.path.exists(label_src):
                                label_dst = os.path.join(lacuna_yolo_path, 
                                                        Path(new_name).stem + '.txt')
                                try:
                                    shutil.copy2(label_src, label_dst)
                                    total_annotations += 1
                                except:
                                    pass
                    except:
                        pass
        else:
            print(f"Warning: Thin_Uganda/images not found at {images_path}")
        
        csv_file = os.path.join(thin_uganda_path, "Labels-CSV.csv")
        if os.path.exists(csv_file):
            try:
                shutil.copy2(csv_file, os.path.join(lacuna_path, 'uganda_metadata.csv'))
                print(f"  Copied Uganda metadata CSV")
            except:
                pass
    else:
        print(f"Warning: Thin_Uganda not found at {thin_uganda_path}")
    
    shutil.rmtree(temp_extract, ignore_errors=True)
    
    final_images = len([f for f in os.listdir(lacuna_path) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    final_annotations = len([f for f in os.listdir(lacuna_yolo_path) 
                            if f.endswith('.txt')])
    
    print()
    print("=" * 80)
    print("LACUNA DATASET READY!")
    print("=" * 80)
    print()
    print("FINAL RESULTS:")
    print(f"  Total images: {final_images:,}")
    print(f"  YOLO annotations: {final_annotations:,}")
    if final_images > 0:
        print(f"  Coverage: {(final_annotations/final_images*100):.1f}% annotated")
    
    print()
    print("DATASET BREAKDOWN:")
    print(f"  Ghana (Thick):  {ghana_thick_count:,} images")
    print(f"  Ghana (Thin):   {ghana_thin_count:,} images")
    print(f"  Uganda (Thin):  {uganda_thin_count:,} images")
    print(f"  Total Ghana:    {ghana_thick_count + ghana_thin_count:,}")
    print(f"  Total Uganda:   {uganda_thin_count:,}")
    
    print()
    print("SMEAR TYPES:")
    print(f"  Thick smears: {ghana_thick_count:,}")
    print(f"  Thin smears:  {ghana_thin_count + uganda_thin_count:,}")
    
    print()
    print("USE CASES:")
    print("  - Binary classification (infected vs uninfected)")
    print("  - Object detection (parasite localization)")
    print("  - Parasitemia estimation")
    print("  - Multi-stage parasite detection")
    
    print()
    print("LOCATIONS:")
    print(f"  Images:      {lacuna_path}")
    print(f"  YOLO Labels: {lacuna_yolo_path}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Troubleshooting:")
    print("  1. Check if Google Drive link is publicly accessible")
    print("  2. Verify file ID is correct")
    print("  3. Check internet connection")
    print("  4. Try re-running this cell")

print()
print("=" * 80)
print("Dataset 5/6 complete!")
print("=" * 80)
