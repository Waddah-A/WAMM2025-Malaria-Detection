# WAMM2025 - Malaria Detection Model

A deep learning pipeline for automated malaria parasite detection from blood smear microscopy images using MaxViT-Small architecture.

## Overview

WAMM2025 is a binary classification model that detects malaria parasites in blood smear images. The pipeline includes comprehensive data preprocessing, training with state-of-the-art augmentation, and multiple explainability methods for model interpretation.

## Results

**Test Set Performance (n=5,691 images):**

| Metric | Value |
|--------|-------|
| Accuracy | 95.85% |
| Precision | 97.48% |
| Recall | 95.61% |
| Specificity | 96.22% |
| F1 Score | 96.54% |
| AUC-ROC | 0.9897 |

**Confusion Matrix:**
- True Negatives: 2,166
- False Positives: 85 (3.78%)
- False Negatives: 151 (4.39%)
- True Positives: 3,289

**Training Details:**
- Best model selected at Epoch 8 (validation AUC: 0.9921)
- Training time: 2.07 hours
- Train-validation gap: 1.78% (robust generalization)
- Perfect validation-test concordance (95.85%)

## Model Architecture

- **Architecture**: MaxViT-Small (CNN-Transformer hybrid)
- **Input Resolution**: 384×384 pixels
- **Classes**: 2 (Parasitized, Uninfected)
- **Framework**: PyTorch with timm

## Dataset Sources

The training pipeline supports 6 diverse datasets (~38,000 images total):

| Dataset | Images | Source | Notes |
|---------|--------|--------|-------|
| NIH | 27,558 | NIH/NLM | Primary dataset |
| Lacuna | ~3,314 | Harvard Dataverse | Ghana + Uganda, includes YOLO annotations |
| Tanzania (NM-AIST) | 3,544 | Harvard Dataverse | 4K resolution |
| Broad Institute | 1,364 | Kaggle | BBBC041 |
| Tek et al. | 655 | Manual | Bounding box annotations |
| MP-IDB | 229 | GitHub | 4 Plasmodium species |

## Installation

```bash
# Clone repository
git clone https://github.com/Waddah-A/WAMM2025-Malaria-Detection.git
cd WAMM2025-Malaria-Detection

# Install dependencies (in Google Colab or local environment)
pip install -r requirements.txt
```

## Usage

Run scripts in order on Google Colab (GPU recommended):

### 1. Setup
```python
# Run code_01_setup.py - Installs packages and verifies GPU
```

### 2. Download Datasets
```python
# Run in sequence:
# code_02_dataset_nih.py      - NIH dataset (auto-download)
# code_03_dataset_mpidb.py    - MP-IDB dataset (auto-clone)
# code_04_dataset_broad.py    - Broad Institute (requires Kaggle API)
# code_05_dataset_tek.py      - Tek dataset (manual upload)
# code_05a_install_rar.py     - Install RAR tools (if needed)
# code_06_dataset_lacuna.py   - Lacuna dataset (auto-download)
# code_07_dataset_tanzania.py - Tanzania dataset (auto-download)
```

### 3. Prepare Data
```python
# code_08_verify_datasets.py  - Verify all datasets
# code_09_organize.py         - Combine into unified structure
# code_10_fix_corrupted.py    - Remove corrupted images
```

### 4. Train Model
```python
# code_11_train.py - Train MaxViT-Small (A100-optimized)
```

### 5. Generate Explanations
```python
# code_12_explainability_main.py - Generate visualization maps
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Max Epochs | 40 |
| Early Stopping | 5 epochs |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts |
| Split | 70/15/15 (train/val/test) |

## Preprocessing Pipeline

1. **Background Normalization**: CLAHE on L channel (LAB color space)
2. **Color Constancy**: Gray world algorithm
3. **Stain Normalization**: Macenko method

## Data Augmentation (Kornia GPU)

- RandomResizedCrop (scale 0.85-1.0)
- RandomRotation (±15°)
- RandomHorizontalFlip
- RandomVerticalFlip
- ColorJitter
- RandomGaussianBlur

## Explainability Methods

The pipeline includes 5 state-of-the-art visualization methods:

1. **HiResCAM** (2023) - Highest spatial accuracy
2. **Integrated Gradients** - Axiomatically rigorous
3. **Grad-CAM++** - Improved localization
4. **Score-CAM** - Gradient-free
5. **Attention Rollout** - Transformer attention visualization


## Output Files

After training:
- `WAMM2025_best_model_preaug.pth` - Trained model weights
- `WAMM2025_confusion_matrix.png` - Test set confusion matrix
- `WAMM2025_training_curves.png` - Training/validation metrics
- `WAMM2025_history.json` - Complete training history

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (A100 recommended)
- 16GB+ GPU memory for batch size 64

## File Structure

```
WAMM2025/
├── code_01_setup.py              # Environment setup
├── code_02_dataset_nih.py        # NIH dataset download
├── code_03_dataset_mpidb.py      # MP-IDB dataset download
├── code_04_dataset_broad.py      # Broad Institute download
├── code_05_dataset_tek.py        # Tek dataset upload
├── code_05a_install_rar.py       # RAR tools installation
├── code_06_dataset_lacuna.py     # Lacuna dataset download
├── code_07_dataset_tanzania.py   # Tanzania dataset download
├── code_08_verify_datasets.py    # Dataset verification
├── code_09_organize.py           # Dataset organization
├── code_10_fix_corrupted.py      # Image validation
├── code_11_train.py              # Model training
├── code_12_explainability_main.py# Explainability generation
├── requirements.txt              # Dependencies
└── README.md
```

## Citation

If you use this code, please cite the relevant datasets:

```bibtex
@misc{nih_malaria,
  title={Malaria Datasets},
  author={NIH/NLM},
  url={https://data.lhncbc.nlm.nih.gov/public/Malaria/}
}

@article{lacuna_malaria,
  title={Lacuna Malaria Dataset},
  doi={10.7910/DVN/VEADSE}
}

@article{tanzania_malaria,
  title={NM-AIST Tanzania Malaria Dataset},
  doi={10.7910/DVN/O2WVWA}
}
```

## Limitations

**Stain Normalization**: This implementation uses Macenko stain normalization, 
which may have negatively impacted model performance. Stain normalization can 
alter color distributions and potentially reduce discriminative features that 
the model could otherwise learn from.

The model achieved 95.85% accuracy, but higher performance may be possible with:
- No stain normalization (let the model learn color invariance)
- Alternative normalization methods (Reinhard, Vahadane, or learned approaches)
- Dataset-specific normalization parameters

**I am open to suggestions for alternative normalization methods or preprocessing 
techniques that could improve performance. Please open an issue or submit a pull 
request if you have recommendations.**


## Acknowledgments

- **AI Assistance**: Code development assisted by Claude (Anthropic)
- **Datasets**: NIH/NLM, Harvard Dataverse (Lacuna, Tanzania), Broad Institute, Kaggle community
- **Libraries**: PyTorch, timm, Kornia