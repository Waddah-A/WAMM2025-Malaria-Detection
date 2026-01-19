"""
WAMM2025 - Malaria Detection System
Dataset 3: Broad Institute BBBC041

1,364 high-quality images (~80,000 cells)
Source: Kaggle - https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes
Paper: Hung et al., Applying Faster R-CNN for Object Detection on Malaria Images
"""

from google.colab import files
import os
import shutil

print("=" * 80)
print("DATASET 3/6: BROAD INSTITUTE")
print("1,364 images via Kaggle")
print("=" * 80)

broad_path = "/content/broad_extracted"

if os.path.exists(broad_path):
    print("Broad Institute dataset already downloaded!")

    img_count = sum([len([f for f in files_list if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
                     for _, _, files_list in os.walk(broad_path)])
    print(f"  Images: {img_count:,}")
else:
    print("KAGGLE SETUP REQUIRED")
    print("=" * 80)
    print()
    print("Steps to get your Kaggle API token:")
    print("  1. Go to: https://www.kaggle.com/settings/account")
    print("  2. Scroll to 'API' section")
    print("  3. Click 'Create New Token'")
    print("  4. This downloads 'kaggle.json'")
    print("  5. Upload it below")
    print()
    print("Upload your kaggle.json file now:")

    try:
        uploaded = files.upload()

        if 'kaggle.json' in uploaded:
            print("kaggle.json received!")

            os.makedirs('/root/.kaggle', exist_ok=True)
            shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
            os.chmod('/root/.kaggle/kaggle.json', 0o600)

            print("Kaggle credentials configured!")

            print()
            print("Downloading Broad Institute dataset from Kaggle...")
            print("  Dataset: malaria-bounding-boxes")
            print("  Size: ~50 MB")
            print("  Time: ~1 minute")

            os.makedirs(broad_path, exist_ok=True)

            os.system('kaggle datasets download -d kmader/malaria-bounding-boxes -p /content/broad_temp --unzip')

            if os.path.exists("/content/broad_temp"):
                for item in os.listdir("/content/broad_temp"):
                    src = f"/content/broad_temp/{item}"
                    dst = f"{broad_path}/{item}"

                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

                shutil.rmtree("/content/broad_temp", ignore_errors=True)

                img_count = sum([len([f for f in files_list if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
                                 for _, _, files_list in os.walk(broad_path)])

                json_count = sum([len([f for f in files_list if f.endswith('.json')])
                                  for _, _, files_list in os.walk(broad_path)])

                print(f"BROAD INSTITUTE DATASET READY!")
                print(f"  Images: {img_count:,}")
                print(f"  JSON annotations: {json_count}")
                print(f"  Citation: Hung et al., 2018 (Faster R-CNN paper)")

                if img_count < 1000:
                    print(f"Warning: Expected ~1,364 images")
                    print(f"Dataset may need structure adjustment")
            else:
                print("Download failed - files not found")
        else:
            print("kaggle.json not uploaded")
            print("You can skip this dataset if needed")

    except KeyboardInterrupt:
        print("Skipped - you can continue without Broad dataset")

    except Exception as e:
        print(f"Error: {e}")
        print("You can skip this dataset and proceed")

print()
print("=" * 80)
print("Dataset 3/6 complete!")
print("=" * 80)
