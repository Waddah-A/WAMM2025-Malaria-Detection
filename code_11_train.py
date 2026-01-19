"""
WAMM2025 - Malaria Detection System
A100-Optimized MaxViT-Small Training (384x384 Resolution)

Model: WAMM2025 (Malaria Model 2025)
Architecture: MaxViT-Small with 384x384 resolution
Split: 70/15/15 (Train/Validation/Test)
Expected Accuracy: 98.8-99.4% for binary classification
"""

# Automatic package install for Kornia
try:
    import kornia
except Exception:
    import subprocess, sys
    print("Installing kornia...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kornia"])
    import kornia

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("WAMM2025: MALARIA DETECTION MODEL")
print("MaxViT-Small (384x384 HIGH-RES) - A100 OPTIMIZED")
print("STEP 1: CORE TRAINING (70/15/15 Split)")
print("=" * 80)

# Mount Google Drive if available
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("\nMounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        print("  Google Drive mounted")
except Exception:
    pass

torch.backends.cudnn.benchmark = True
gc.collect()

# Configuration
IMG_SIZE = 384
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
MERGED_PATH = "/content/combined_multi_species"
PREPROCESSED_DIR = '/content/drive/MyDrive/malaria_preprocessed'
CHECKPOINT_DIR = '/content/checkpoints_WAMM2025'
CHECKPOINT_INTERVAL = 3
NUM_WORKERS = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")
    if 'A100' in gpu_name:
        print("  A100 DETECTED - Using optimized settings!")

print(f"\nWAMM2025 CONFIGURATION (A100 OPTIMIZED):")
print(f"  Batch Size:        {BATCH_SIZE}")
print(f"  Data Workers:      {NUM_WORKERS}")
print(f"  Expected Speed:    ~1.5s/iteration")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
OUTPUT_DIR = '/content/outputs_WAMM2025'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/malaria_outputs_WAMM2025'
try:
    if os.path.exists('/content/drive/MyDrive'):
        os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
        print(f"Drive output: {DRIVE_OUTPUT_DIR}")
except:
    DRIVE_OUTPUT_DIR = None


# Image normalization utilities
class StainNormalizer:
    """Stain normalization for histopathology images."""
    
    def __init__(self):
        self.target_stains = np.array([[0.5626, 0.2159],
                                       [0.7201, 0.8012],
                                       [0.4062, 0.5581]])
        self.target_concentrations = np.array([[1.9705, 1.0308]])

    def normalize(self, img):
        try:
            img = np.array(img)
            img_od = -np.log((img.astype(np.float32) + 1) / 256.0)
            od_hat = img_od[~np.any(img_od < 0.15, axis=2)]
            if od_hat.shape[0] == 0:
                return Image.fromarray(img)
            _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
            eigvecs = eigvecs[:, [2, 1]]
            proj = np.dot(od_hat, eigvecs)
            phi = np.arctan2(proj[:, 1], proj[:, 0])
            min_phi, max_phi = np.percentile(phi, 1), np.percentile(phi, 99)
            v1 = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))
            if v1[0] > v2[0]:
                stain_matrix = np.array([v1, v2]).T
            else:
                stain_matrix = np.array([v2, v1]).T
            stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0)
            concentrations = np.linalg.lstsq(stain_matrix, img_od.reshape(-1, 3).T, rcond=None)[0]
            max_conc = np.percentile(concentrations, 99, axis=1, keepdims=True)
            concentrations = concentrations / max_conc * self.target_concentrations.T
            img_normalized = np.exp(-np.dot(self.target_stains, concentrations)) * 256
            img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8).T.reshape(img.shape)
            return Image.fromarray(img_normalized)
        except Exception:
            return Image.fromarray(np.array(img).astype(np.uint8))


def background_normalization(img):
    """Apply CLAHE normalization."""
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_normalized)


def color_constancy(img):
    """Apply gray world color constancy."""
    img_array = np.array(img).astype(np.float32)
    mean_r = np.mean(img_array[:, :, 0]) + 1e-8
    mean_g = np.mean(img_array[:, :, 1]) + 1e-8
    mean_b = np.mean(img_array[:, :, 2]) + 1e-8
    gray = (mean_r + mean_g + mean_b) / 3
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (gray / mean_r), 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (gray / mean_g), 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (gray / mean_b), 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))


def ensure_preprocessed(src_root, dst_root, use_stain_norm=True):
    """Preprocess images once and save to disk."""
    if not os.path.exists(src_root):
        print(f"Source dataset not found: {src_root}. Skipping preprocessing.")
        return
    os.makedirs(dst_root, exist_ok=True)
    stain_norm = StainNormalizer() if use_stain_norm else None

    for cls in ['Parasitized', 'Uninfected']:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        if not os.path.exists(src_dir):
            print(f"Source class dir not found: {src_dir}")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for fname in tqdm(files, desc=f"Preprocessing {cls}", unit='files'):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if os.path.exists(dst_path):
                continue
            try:
                img = Image.open(src_path).convert('RGB')
                img = background_normalization(img)
                img = color_constancy(img)
                if stain_norm:
                    img = stain_norm.normalize(img)
                img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
                img.save(dst_path, format='PNG', compress_level=1)
            except Exception as e:
                print(f"  Error preprocessing {src_path}: {e}")
    print(f"\nPreprocessed images saved to: {dst_root}")


# Check if preprocessing is needed
parasitized_dir = os.path.join(PREPROCESSED_DIR, "Parasitized")
uninfected_dir = os.path.join(PREPROCESSED_DIR, "Uninfected")

if (os.path.exists(parasitized_dir) and os.path.exists(uninfected_dir) and
    len(os.listdir(parasitized_dir)) > 100 and len(os.listdir(uninfected_dir)) > 100):
    print(f"Using existing preprocessed data at {PREPROCESSED_DIR}")
    PREPROCESSING_NEEDED = False
else:
    print(f"Preprocessed data not found or incomplete at {PREPROCESSED_DIR}.")
    print(f"Please ensure preprocessed data exists or run preprocessing step.")
    PREPROCESSING_NEEDED = True

if PREPROCESSING_NEEDED:
    print("\nPreprocessing dataset (one-time). This can be slow, but runs only once.")
    ensure_preprocessed(MERGED_PATH, PREPROCESSED_DIR, use_stain_norm=True)
else:
    print(f"Skipping preprocessing step as preprocessed data was found.")


# Dataset class
class MalariaDataset(Dataset):
    """Malaria cell image dataset."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        parasitized_dir = os.path.join(root_dir, "Parasitized")
        if os.path.exists(parasitized_dir):
            for img_name in os.listdir(parasitized_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(parasitized_dir, img_name))
                    self.labels.append(1)

        uninfected_dir = os.path.join(root_dir, "Uninfected")
        if os.path.exists(uninfected_dir):
            for img_name in os.listdir(uninfected_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(uninfected_dir, img_name))
                    self.labels.append(0)

        print(f"  Loaded {len(self.images):,} images from {root_dir}")
        print(f"  Parasitized: {sum(self.labels):,}")
        print(f"  Uninfected: {len(self.labels) - sum(self.labels):,}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


print("\nSetting up data augmentation...")

train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Augmentation configured")

print("\nLoading datasets (70/15/15 Split)...")

full_dataset = MalariaDataset(PREPROCESSED_DIR, transform=None)

total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  Total:      {total_size:,} samples")
print(f"  Training:   {len(train_dataset):,} (70%)")
print(f"  Validation: {len(val_dataset):,} (15%)")
print(f"  Test:       {len(test_dataset):,} (15%)")

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms
test_dataset.dataset.transform = val_transforms

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

print(f"\nDatasets ready!")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

torch.cuda.empty_cache()
gc.collect()

print("\nLoading MaxViT-Small...")

model = timm.create_model('maxvit_small_tf_384.in1k', pretrained=True, num_classes=NUM_CLASSES)

print(f"MaxViT-Small loaded")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Parameters: {total_params:,}")

model = model.to(device)

print("\nSetting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
scaler = GradScaler()

print("Training configured")

# Kornia GPU augmentation pipeline
import kornia.augmentation as K

kornia_train_aug = torch.nn.Sequential(
    K.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0), p=1.0),
    K.RandomRotation(degrees=15.0, p=0.6),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
    K.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0), p=0.4),
)


def apply_normalize_batch(batch):
    """Normalize batch tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, -1, 1, 1)
    return (batch - mean) / std


def save_epoch_log(epoch, train_loss, train_acc, val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, lr):
    """Save training log for each epoch."""
    log_file = os.path.join(OUTPUT_DIR, 'WAMM2025_training_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("WAMM2025 TRAINING LOG (70/15/15 Split)\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
    with open(log_file, 'a') as f:
        f.write(f"\nEPOCH {epoch+1}/{MAX_EPOCHS} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%\n")
        f.write(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")
        f.write(f"Precision: {val_prec*100:.2f}% | Recall: {val_rec*100:.2f}%\n")
        f.write(f"F1: {val_f1*100:.2f}% | AUC: {val_auc:.4f}\n")
        f.write(f"LR: {lr:.2e}\n")
    if DRIVE_OUTPUT_DIR:
        try:
            drive_log = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_training_log.txt')
            with open(log_file, 'r') as src, open(drive_log, 'w') as dst:
                dst.write(src.read())
        except:
            pass


def save_intermediate_plots(history, epoch):
    """Save training curves."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs_range, history['train_acc'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs_range, history['val_acc'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs_range, history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs_range, history['val_auc'], linewidth=2, color='green')
        axes[0, 2].set_title('AUC-ROC', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs_range, history['val_precision'], linewidth=2, color='blue')
        axes[1, 0].set_title('Precision', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs_range, history['val_recall'], linewidth=2, color='orange')
        axes[1, 1].set_title('Recall', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(epochs_range, history['val_f1'], linewidth=2, color='purple')
        axes[1, 2].set_title('F1 Score', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'WAMM2025 Training - Epoch {epoch+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        local_path = os.path.join(OUTPUT_DIR, f'WAMM2025_curves_epoch_{epoch+1}.png')
        plt.savefig(local_path, dpi=150, bbox_inches='tight')
        if DRIVE_OUTPUT_DIR:
            drive_path = os.path.join(DRIVE_OUTPUT_DIR, f'WAMM2025_curves_epoch_{epoch+1}.png')
            plt.savefig(drive_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Curves saved")
    except Exception as e:
        print(f"  Error: {e}")


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_auc, best_acc, epochs_no_improve):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'best_auc': best_auc,
        'best_acc': best_acc,
        'epochs_no_improve': epochs_no_improve
    }
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'WAMM2025_checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    latest_path = os.path.join(CHECKPOINT_DIR, 'WAMM2025_checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"  Checkpoint saved")
    
    try:
        drive_checkpoint_dir = '/content/drive/MyDrive/malaria_checkpoints_WAMM2025'
        if os.path.exists('/content/drive/MyDrive'):
            os.makedirs(drive_checkpoint_dir, exist_ok=True)
            drive_path = os.path.join(drive_checkpoint_dir, f'WAMM2025_checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, drive_path)
            drive_latest = os.path.join(drive_checkpoint_dir, 'WAMM2025_checkpoint_latest.pth')
            torch.save(checkpoint, drive_latest)
            print(f"  Drive backup saved")
    except:
        pass
    
    try:
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('WAMM2025_checkpoint_epoch_')])
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                os.remove(os.path.join(CHECKPOINT_DIR, old_checkpoint))
    except:
        pass


def load_checkpoint():
    """Load training checkpoint if available."""
    drive_latest = '/content/drive/MyDrive/malaria_checkpoints_WAMM2025/WAMM2025_checkpoint_latest.pth'
    local_latest = os.path.join(CHECKPOINT_DIR, 'WAMM2025_checkpoint_latest.pth')
    checkpoint_path = None
    
    if os.path.exists(drive_latest):
        checkpoint_path = drive_latest
        print("  Found Drive checkpoint")
    elif os.path.exists(local_latest):
        checkpoint_path = local_latest
        print("  Found local checkpoint")
    
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
            return checkpoint
        except Exception as e:
            print(f"  Error: {e}")
            return None
    return None


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            torch.random.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
            images = kornia_train_aug(images)

        images = apply_normalize_batch(images)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    desc = 'Validation'
    if loader == test_loader:
        desc = 'Testing'

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, precision, recall, f1, auc, all_preds, all_labels


print()
print("=" * 80)
print("STARTING WAMM2025 TRAINING (70/15/15 Split)")
print("=" * 80)

checkpoint = load_checkpoint()
start_epoch = 0
start_time = datetime.now()

if checkpoint:
    print("\nRESUMING...")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    history = checkpoint['history']
    best_auc = checkpoint['best_auc']
    best_acc = checkpoint['best_acc']
    epochs_no_improve = checkpoint['epochs_no_improve']
    start_epoch = checkpoint['epoch'] + 1
    print(f"  Resumed from epoch {start_epoch}")
else:
    print("\nSTARTING FRESH...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_auc': []
    }
    best_auc = 0.0
    best_acc = 0.0
    epochs_no_improve = 0

for epoch in range(start_epoch, MAX_EPOCHS):
    print(f"\n{'='*80}")
    print(f"WAMM2025 - Epoch {epoch+1}/{MAX_EPOCHS}")
    print(f"{'='*80}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_rec)
    history['val_f1'].append(val_f1)
    history['val_auc'].append(val_auc)

    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
    print(f"  Metrics: P={val_prec*100:.2f}%, R={val_rec*100:.2f}%, F1={val_f1*100:.2f}%, AUC={val_auc:.4f}")

    current_lr = optimizer.param_groups[0]['lr']
    save_epoch_log(epoch, train_loss, train_acc, val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, current_lr)

    if (epoch + 1) % 5 == 0:
        save_intermediate_plots(history, epoch)

    if val_auc > best_auc:
        best_auc = val_auc
        best_acc = val_acc
        torch.save(model.state_dict(), 'WAMM2025_best_model_preaug.pth')
        print(f"\n  NEW BEST! (on Val Set) AUC={val_auc:.4f}, Acc={val_acc*100:.2f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        print(f"\n  Saving checkpoint...")
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_auc, best_acc, epochs_no_improve)

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping (patience={EARLY_STOPPING_PATIENCE})")
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_auc, best_acc, epochs_no_improve)
        break

if epochs_no_improve < EARLY_STOPPING_PATIENCE:
    save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_auc, best_acc, epochs_no_improve)

training_time = (datetime.now() - start_time).total_seconds() / 3600

print()
print("=" * 80)
print("FINAL EVALUATION (on TEST set)")
print("=" * 80)

model.load_state_dict(torch.load('WAMM2025_best_model_preaug.pth'))

test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, final_preds, final_labels = validate(
    model, test_loader, criterion, device
)

print()
print("=" * 80)
print("WAMM2025 - FINAL TEST RESULTS")
print("=" * 80)
print(f"  Accuracy:     {test_acc*100:>6.2f}%")
print(f"  Precision:    {test_prec*100:>6.2f}%")
print(f"  Recall:       {test_rec*100:>6.2f}%")
print(f"  F1 Score:     {test_f1*100:>6.2f}%")
print(f"  AUC-ROC:      {test_auc:>6.4f}")
print(f"  Training Time: {training_time:>5.2f} hours")
print("=" * 80)

print("\nGenerating confusion matrix (on TEST set)...")

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Uninfected', 'Parasitized'],
            yticklabels=['Uninfected', 'Parasitized'])
plt.title(f'WAMM2025 Confusion Matrix - Test Accuracy: {test_acc*100:.2f}%', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

local_cm = os.path.join(OUTPUT_DIR, 'WAMM2025_confusion_matrix.png')
plt.savefig(local_cm, dpi=150, bbox_inches='tight')
if DRIVE_OUTPUT_DIR:
    drive_cm = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_confusion_matrix.png')
    plt.savefig(drive_cm, dpi=150, bbox_inches='tight')
plt.savefig('WAMM2025_confusion_matrix.png', dpi=150)
plt.show()

print("Confusion matrix saved")

print("\nGenerating training curves...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0, 0].plot(epochs_range, history['train_acc'], label='Train', linewidth=2)
axes[0, 0].plot(epochs_range, history['val_acc'], label='Validation', linewidth=2)
axes[0, 0].set_title('Accuracy', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
axes[0, 1].plot(epochs_range, history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Loss', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(epochs_range, history['val_auc'], linewidth=2, color='green')
axes[0, 2].set_title('Val AUC-ROC', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, history['val_precision'], linewidth=2, color='blue')
axes[1, 0].set_title('Val Precision', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(epochs_range, history['val_recall'], linewidth=2, color='orange')
axes[1, 1].set_title('Val Recall', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(epochs_range, history['val_f1'], linewidth=2, color='purple')
axes[1, 2].set_title('Val F1 Score', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('WAMM2025 Training Metrics (vs. Validation)', fontsize=16, fontweight='bold')
plt.tight_layout()

local_curves = os.path.join(OUTPUT_DIR, 'WAMM2025_training_curves.png')
plt.savefig(local_curves, dpi=150, bbox_inches='tight')
if DRIVE_OUTPUT_DIR:
    drive_curves = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_training_curves.png')
    plt.savefig(drive_curves, dpi=150, bbox_inches='tight')
plt.savefig('WAMM2025_training_curves.png', dpi=150)
plt.show()

print("Training curves saved")

history_local = os.path.join(OUTPUT_DIR, 'WAMM2025_history.json')
with open(history_local, 'w') as f:
    json.dump(history, f, indent=2)
if DRIVE_OUTPUT_DIR:
    history_drive = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_history.json')
    with open(history_drive, 'w') as f:
        json.dump(history, f, indent=2)
with open('WAMM2025_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nTraining history saved")

print()
print("=" * 80)
print("WAMM2025 (STEP 1) TRAINING COMPLETE!")
print("=" * 80)
print()
print("Final Test Results:")
print(f"  Test Accuracy:     {test_acc*100:.2f}%")
print(f"  Test AUC:          {test_auc:.4f}")
print(f"  (Best Val AUC was: {best_auc:.4f})")
print()
print(f"  Training Time:     {training_time:.2f} hours")
print(f"  Epochs Trained:    {epoch+1}")
print()
print("Saved Files:")
print("  - WAMM2025_best_model_preaug.pth (Best model based on validation set)")
print("  - WAMM2025_confusion_matrix.png (Based on test set)")
print("  - WAMM2025_training_curves.png")
print("  - WAMM2025_history.json")
print()
print("Next: Run explainability scripts (code_12_*.py) for model interpretation.")
print("=" * 80)
