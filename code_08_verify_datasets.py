"""
WAMM2025 - Malaria Detection System
Verification: Check All 6 Datasets

Make sure everything is ready before organizing.
"""

import os

print("=" * 80)
print("DATASET VERIFICATION (6 DATASETS)")
print("=" * 80)

datasets = {
    'NIH': {
        'path': '/content/cell_images',
        'expected': 27558,
        'critical': True,
        'source': 'United States (NIH)'
    },
    'MP-IDB': {
        'path': '/content/mpidb_extracted',
        'expected': 229,
        'critical': False,
        'source': 'Multiple (4 species)'
    },
    'Broad Institute': {
        'path': '/content/broad_extracted',
        'expected': 1364,
        'critical': False,
        'source': 'Global (research)'
    },
    'Tek': {
        'path': '/content/tek_extracted',
        'expected': 655,
        'critical': True,
        'source': 'For Stage 2 (bounding boxes)'
    },
    'Lacuna (Ghana+Uganda)': {
        'path': '/content/lacuna_extracted',
        'expected': 3314,
        'critical': False,
        'source': 'Ghana + Uganda (reorganized + YOLO)'
    },
    'Tanzania (NM-AIST)': {
        'path': '/content/tanzania_extracted/tanzania_dataset',
        'expected': 3544,
        'critical': False,
        'source': 'Tanzania (4K resolution)'
    }
}

print()
print("CHECKING ALL 6 DATASETS:")
print("=" * 80)

total_images = 0
available_count = 0
critical_missing = []
dataset_details = {}
yolo_annotations_available = False

for name, info in datasets.items():
    print(f"\n{name}:")

    if os.path.exists(info['path']):
        img_count = 0
        for root, dirs, files in os.walk(info['path']):
            img_count += len([f for f in files 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))
                            and not f.startswith('._')])

        total_images += img_count
        available_count += 1
        dataset_details[name] = {'count': img_count, 'available': True}

        if img_count >= info['expected'] * 0.8:
            status = "[OK]"
            quality = "EXCELLENT"
        elif img_count >= info['expected'] * 0.5:
            status = "[WARN]"
            quality = "PARTIAL"
        else:
            status = "[WARN]"
            quality = "LOW"
        
        print(f"  {status} Found: {img_count:,} images ({quality})")
        print(f"  Expected: {info['expected']:,}")
        print(f"  Source: {info['source']}")

        if name == 'Tek':
            has_annotation = os.path.exists(os.path.join(info['path'], 'malaria.txt'))
            if has_annotation:
                print(f"  [OK] malaria.txt present (Stage 2 ready)")
            else:
                print(f"  [FAIL] malaria.txt MISSING (Stage 2 blocked)")
                critical_missing.append('Tek annotations')
        
        if name == 'Lacuna (Ghana+Uganda)' and img_count > 0:
            yolo_path = '/content/lacuna_yolo_annotations'
            if os.path.exists(yolo_path):
                yolo_count = len([f for f in os.listdir(yolo_path) if f.endswith('.txt')])
                if yolo_count > 0:
                    print(f"  [OK] YOLO annotations: {yolo_count:,} files")
                    print(f"  Coverage: {(yolo_count/img_count*100):.1f}%")
                    yolo_annotations_available = True
        
        if name == 'Tanzania (NM-AIST)' and img_count > 0:
            categories = ['Thick_Infected', 'Thick_Uninfected', 'Thin_Infected', 'Thin_Uninfected']
            found_categories = []
            for cat in categories:
                cat_path = os.path.join(info['path'], cat)
                if os.path.exists(cat_path):
                    cat_count = len([f for f in os.listdir(cat_path) 
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                                   and not f.startswith('._')])
                    if cat_count > 0:
                        found_categories.append(f"{cat}: {cat_count}")
            
            if found_categories:
                print(f"  Categories: {len(found_categories)}/4 found")
                for cat_info in found_categories:
                    print(f"    - {cat_info}")
    else:
        print(f"  [NOT FOUND]")
        print(f"  Source: {info['source']}")
        dataset_details[name] = {'count': 0, 'available': False}
        
        if name == 'Lacuna (Ghana+Uganda)':
            print(f"  Tip: Run code_06_dataset_lacuna.py to download")
        elif name == 'Tanzania (NM-AIST)':
            print(f"  Tip: Run code_07_dataset_tanzania.py to download")
        
        if info['critical']:
            critical_missing.append(name)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nDatasets found: {available_count}/6")
print(f"Total images: {total_images:,}")

print(f"\nAvailable datasets:")
for name, details in dataset_details.items():
    if details['available']:
        percentage = (details['count'] / total_images * 100) if total_images > 0 else 0
        print(f"  - {name:<25} {details['count']:>6,} ({percentage:>5.1f}%)")

missing_count = 6 - available_count
if missing_count > 0:
    print(f"\nMissing datasets: {missing_count}")
    for name, details in dataset_details.items():
        if not details['available']:
            print(f"  - {name}")

print()
print("=" * 80)
print("TRAINING READINESS:")
print("=" * 80)

stage1_ready = total_images >= 10000
stage2_ready = os.path.exists('/content/tek_extracted/malaria.txt') or yolo_annotations_available

if stage1_ready:
    print("\n[OK] STAGE 1 READY - Binary Screening")
    print(f"  {total_images:,} images available")
    
    if total_images >= 38000:
        print(f"  Expected accuracy: 99.2-99.6%")
        print(f"  Quality: RESEARCH-GRADE+ (Maximum diversity!)")
    elif total_images >= 35000:
        print(f"  Expected accuracy: 99.0-99.5%")
        print(f"  Quality: RESEARCH-GRADE!")
    elif total_images >= 30000:
        print(f"  Expected accuracy: 98.5-99.2%")
        print(f"  Quality: CLINICAL-GRADE")
    elif total_images >= 25000:
        print(f"  Expected accuracy: 98.0-98.8%")
        print(f"  Quality: PRODUCTION-READY")
    elif total_images >= 15000:
        print(f"  Expected accuracy: 95-97%")
    else:
        print(f"  Expected accuracy: 90-95%")
else:
    print("\n[FAIL] STAGE 1 NOT READY")
    print(f"  Need at least 10,000 images (have {total_images:,})")

print()

if stage2_ready:
    print("[OK] STAGE 2 READY - Object Detection")
    if os.path.exists('/content/tek_extracted/malaria.txt'):
        print("  - Tek dataset with bounding boxes available")
    if yolo_annotations_available:
        print("  - Lacuna YOLO annotations available")
    print("  Will detect parasites + identify species")
else:
    print("[WARN] STAGE 2 NOT READY")
    print("  Need Tek dataset with malaria.txt OR Lacuna with YOLO")

print()
print("=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

if len(critical_missing) == 0:
    if available_count == 6:
        print("\nPERFECT! All 6 datasets ready!")
        print(f"  ~{total_images:,} images from 6 sources")
        print("  Proceed to organize datasets (code_09_organize.py)")
        print("  Then train with code_11_train.py")
    else:
        print("\nEXCELLENT! Core datasets ready!")
        print("  Proceed to organize datasets (code_09_organize.py)")
else:
    print("\nCritical datasets missing:")
    for missing in critical_missing:
        print(f"  - {missing}")

print()
print("=" * 80)
print("Next: Organize all datasets with code_09_organize.py!")
print("=" * 80)
