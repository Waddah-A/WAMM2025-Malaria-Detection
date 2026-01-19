"""
WAMM2025 - Malaria Detection System
Dataset 4: Tek et al. Dataset

655 images with bounding box annotations
CRITICAL FOR STAGE 2 OBJECT DETECTION!

Source: https://github.com/tobsecret/Awesome_Malaria_Parasite_Imaging_Datasets
Paper: Tek et al., 2016 - Parasite detection and identification
"""

from google.colab import files
import zipfile
import os
import shutil

print("=" * 80)
print("DATASET 4/6: TEK DATASET")
print("655 images + bounding boxes")
print("CRITICAL FOR STAGE 2!")
print("=" * 80)

tek_path = "/content/tek_extracted"

if os.path.exists(tek_path):
    images = [f for f in os.listdir(tek_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    has_annotation = os.path.exists(os.path.join(tek_path, 'malaria.txt'))

    print(f"Tek dataset already present!")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {'malaria.txt found' if has_annotation else 'malaria.txt MISSING'}")

    if len(images) >= 500 and has_annotation:
        print(f"TEK DATASET COMPLETE - Stage 2 ready!")
    elif not has_annotation:
        print(f"Need malaria.txt for Stage 2 object detection")

else:
    os.makedirs(tek_path, exist_ok=True)

    print("BEFORE UPLOADING:")
    print("=" * 80)
    print()
    print("Download Tek dataset:")
    print("  1. Go to: https://github.com/tobsecret/Awesome_Malaria_Parasite_Imaging_Datasets")
    print("  2. Find 'Tek et al. 2016' section")
    print("  3. Download TWO things:")
    print("     - 655 images (thin blood smears)")
    print("     - malaria.txt (annotation file)")
    print()
    print("Create a ZIP file:")
    print("  - Put BOTH images and malaria.txt in one ZIP")
    print("  - Name it: tek_dataset.zip")
    print()
    print("=" * 80)
    print()
    print("UPLOAD YOUR TEK DATASET ZIP FILE:")

    try:
        uploaded = files.upload()

        for filename in uploaded.keys():
            print(f"Uploaded: {filename}")
            print(f"Extracting...")

            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(tek_path)

                os.remove(filename)

                images = [f for f in os.listdir(tek_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                has_annotation = os.path.exists(os.path.join(tek_path, 'malaria.txt'))

                if len(images) == 0:
                    for root, dirs, files_list in os.walk(tek_path):
                        for f in files_list:
                            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                                shutil.move(os.path.join(root, f), os.path.join(tek_path, f))
                            elif f == 'malaria.txt':
                                shutil.move(os.path.join(root, f), os.path.join(tek_path, 'malaria.txt'))

                    images = [f for f in os.listdir(tek_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    has_annotation = os.path.exists(os.path.join(tek_path, 'malaria.txt'))

                print(f"EXTRACTION COMPLETE!")
                print(f"  Images: {len(images)}")
                print(f"  malaria.txt: {'Found' if has_annotation else 'Missing'}")

                if len(images) >= 500 and has_annotation:
                    print(f"TEK DATASET READY FOR STAGE 2!")

                    with open(os.path.join(tek_path, 'malaria.txt'), 'r') as f:
                        lines = [l for l in f if l.strip()]
                    print(f"  Annotations: {len(lines)} parasite instances")

                elif len(images) < 500:
                    print(f"Expected ~655 images, found {len(images)}")
                    print(f"Check if all images were in the ZIP")

                if not has_annotation:
                    print(f"CRITICAL: malaria.txt not found!")
                    print(f"Stage 2 requires this annotation file")
                    print(f"Please upload it separately or re-upload ZIP")

            except zipfile.BadZipFile:
                print(f"Error: {filename} is not a valid ZIP file")
                print(f"Please create a proper ZIP and try again")

            except Exception as e:
                print(f"Error: {e}")
                print(f"Try re-uploading the ZIP file")

    except KeyboardInterrupt:
        print("Upload cancelled")
        print("Tek dataset is REQUIRED for Stage 2")
        print("You can upload it later before Stage 2 training")

print()
print("=" * 80)
print("Dataset 4/6 upload complete!")
print("=" * 80)
