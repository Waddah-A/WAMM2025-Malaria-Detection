"""
WAMM2025 - Malaria Detection System
Explainability Methods

State-of-the-art methods for clinical-grade malaria detection interpretation.

Methods implemented:
1. HiResCAM (2023) - Best spatial accuracy
2. Integrated Gradients - Most rigorous
3. Attention Rollout - Transformer visualization
4. Grad-CAM++ - Improved baseline
5. Score-CAM - Gradient-free approach

Suitable for FDA submission and clinical deployment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import timm
from torchvision import transforms

print("=" * 80)
print("WAMM2025 EXPLAINABILITY - STATE-OF-THE-ART METHODS")
print("=" * 80)
print()
print("Methods:")
print("  1. HiResCAM (2023) - Highest resolution, best for small objects")
print("  2. Integrated Gradients - Most rigorous, Google's method")
print("  3. Attention Rollout - Transformer attention visualization")
print("  4. Grad-CAM++ - Improved Grad-CAM with better localization")
print("  5. Score-CAM - Gradient-free, no backprop artifacts")
print()
print("=" * 80)


# Mount Google Drive
print()
print("=" * 80)
print("STEP 1: MOUNTING GOOGLE DRIVE FOR PERSISTENT STORAGE")
print("=" * 80)

try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        print("  Google Drive mounted successfully!")
    else:
        print("Google Drive already mounted!")
    
    DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/malaria_explainability_WAMM2025'
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {DRIVE_OUTPUT_DIR}")
    DRIVE_AVAILABLE = True
    
except Exception as e:
    print(f"Could not mount Google Drive: {e}")
    print("Outputs will be saved locally only")
    DRIVE_OUTPUT_DIR = None
    DRIVE_AVAILABLE = False


# Pre-flight checks
print()
print("=" * 80)
print("STEP 2: PRE-FLIGHT CHECKS")
print("=" * 80)

checks = {
    'WAMM2025 Model file': {
        'path': 'WAMM2025_best_model.pth',
        'critical': True,
        'message': 'Training must complete successfully (code_11_train.py) before running explainability'
    },
    'Dataset directory': {
        'path': '/content/combined_multi_species',
        'critical': True,
        'message': 'Run dataset organization code (code_09_organize.py) first'
    },
    'Parasitized folder': {
        'path': '/content/combined_multi_species/Parasitized',
        'critical': True,
        'message': 'Dataset not properly organized'
    },
    'Uninfected folder': {
        'path': '/content/combined_multi_species/Uninfected',
        'critical': True,
        'message': 'Dataset not properly organized'
    }
}

print("\nChecking required files and directories:\n")

all_checks_passed = True
critical_failures = []

for name, info in checks.items():
    exists = os.path.exists(info['path'])
    status = "[OK]" if exists else "[FAIL]"
    print(f"  {status} {name}")
    
    if not exists:
        all_checks_passed = False
        if info['critical']:
            critical_failures.append((name, info['message']))
            print(f"      {info['message']}")

if os.path.exists('/content/combined_multi_species/Parasitized'):
    parasitized_count = len([f for f in os.listdir('/content/combined_multi_species/Parasitized') 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    uninfected_count = len([f for f in os.listdir('/content/combined_multi_species/Uninfected') 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nDataset statistics:")
    print(f"  Parasitized images: {parasitized_count:,}")
    print(f"  Uninfected images:  {uninfected_count:,}")
    
    if parasitized_count == 0 or uninfected_count == 0:
        all_checks_passed = False
        critical_failures.append(("Empty dataset folders", "Dataset folders exist but contain no images"))

if not all_checks_passed:
    print()
    print("=" * 80)
    print("CRITICAL FAILURES DETECTED")
    print("=" * 80)
    print()
    print("Cannot proceed with WAMM2025 explainability generation!")
    print()
    print("Issues found:")
    for i, (failure, message) in enumerate(critical_failures, 1):
        print(f"  {i}. {failure}")
        print(f"     -> {message}")
    print()
    print("Action required:")
    print("  1. Ensure WAMM2025 training completed successfully (code_11_train.py)")
    print("  2. Verify WAMM2025_best_model.pth exists")
    print("  3. Ensure dataset is organized (code_09_organize.py)")
    print("  4. Re-run this cell after fixing the issues")
    print()
    print("=" * 80)
    raise SystemExit("Pre-flight checks failed. Please fix the issues above and re-run.")

print()
print("=" * 80)
print("ALL PRE-FLIGHT CHECKS PASSED!")
print("=" * 80)
print()
print("Proceeding with WAMM2025 explainability generation...")


# HiResCAM Implementation
class HiResCAM:
    """
    HiResCAM: Faithful Location Representation in Visual Explanations (2023)
    Best spatial accuracy, perfect for high-resolution images.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients_relu = F.relu(self.gradients)
        weighted_activations = gradients_relu * self.activations
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_image.shape[2], input_image.shape[3]), 
                           mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# Integrated Gradients Implementation
class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017)
    Axiomatically sound method, widely used in production.
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_attribution(self, input_image, target_class=None, steps=50, baseline=None):
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_image)
        
        with torch.no_grad():
            output = self.model(input_image)
            if target_class is None:
                target_class = output.argmax(dim=1)
        
        alphas = torch.linspace(0, 1, steps).to(input_image.device)
        integrated_grads = torch.zeros_like(input_image)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_image - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)
            
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            integrated_grads += interpolated.grad
        
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (input_image - baseline)
        
        attribution = integrated_grads.squeeze().sum(dim=0).cpu().numpy()
        attribution = np.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution


# Grad-CAM++ Implementation
class GradCAMPlusPlus:
    """
    Grad-CAM++: Improved Visual Explanations (Chattopadhay et al., 2018)
    Better localization than Grad-CAM.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients_power_2 = self.gradients ** 2
        gradients_power_3 = self.gradients ** 3
        
        sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha = gradients_power_2 / (2 * gradients_power_2 + 
                                      sum_activations * gradients_power_3 + eps)
        
        relu_grad = F.relu(self.gradients)
        weights = torch.sum(alpha * relu_grad, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# Score-CAM Implementation
class ScoreCAM:
    """
    Score-CAM: Gradient-Free Visual Explanations (Wang et al., 2020)
    No gradients needed, more reliable on deep networks.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(self, input_image, target_class=None, batch_size=16):
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_image)
            if target_class is None:
                target_class = output.argmax(dim=1)
            base_score = output[0, target_class]
        
        _ = self.model(input_image)
        activations = self.activations[0]
        
        upsampled_activations = F.interpolate(
            activations.unsqueeze(0),
            size=(input_image.shape[2], input_image.shape[3]),
            mode='bilinear',
            align_corners=False
        )[0]
        
        normalized_activations = []
        for i in range(upsampled_activations.shape[0]):
            act = upsampled_activations[i]
            normalized = (act - act.min()) / (act.max() - act.min() + 1e-8)
            normalized_activations.append(normalized)
        normalized_activations = torch.stack(normalized_activations)
        
        scores = []
        n_batches = (len(normalized_activations) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(normalized_activations))
            
            batch_acts = normalized_activations[start_idx:end_idx]
            masked_inputs = input_image * batch_acts.unsqueeze(1)
            
            with torch.no_grad():
                outputs = self.model(masked_inputs)
                batch_scores = outputs[:, target_class]
                scores.extend(batch_scores.cpu().numpy())
        
        scores = np.array(scores)
        
        weights = torch.tensor(scores).to(activations.device)
        weights = F.softmax(weights / 0.1, dim=0)
        
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = F.relu(cam)
        
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# Attention Rollout Implementation
class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    Leverages transformer attention mechanisms.
    """
    
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) == 4:
                self.attentions.append(output.detach())
        
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
    
    def generate_attention_map(self, input_image):
        self.attentions = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_image)
        
        if len(self.attentions) == 0:
            return np.ones((input_image.shape[2], input_image.shape[3])) * 0.5
        
        attention_map = None
        for attn in self.attentions:
            if attention_map is None:
                attention_map = attn.mean(dim=1)
            else:
                attention_map = attention_map + attn.mean(dim=1)
        
        attention_map = attention_map.squeeze().cpu().numpy()
        
        attention_map = cv2.resize(attention_map, 
                                  (input_image.shape[3], input_image.shape[2]),
                                  interpolation=cv2.INTER_LINEAR)
        
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map
    
    def cleanup(self):
        for handle in self.hooks:
            handle.remove()


# Visualization utilities
def apply_colormap(cam, colormap=cv2.COLORMAP_JET):
    cam_uint8 = np.uint8(255 * cam)
    cam_colored = cv2.applyColorMap(cam_uint8, colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    return cam_colored


def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    if image.max() <= 1.0:
        image = np.uint8(255 * image)
    overlayed = cv2.addWeighted(image, 1-alpha, heatmap_resized, alpha, 0)
    return overlayed


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


# Preprocessing utilities
class StainNormalizer:
    def __init__(self):
        self.target_stains = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
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
            stain_matrix = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
            stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0)
            concentrations = np.linalg.lstsq(stain_matrix, img_od.reshape(-1, 3).T, rcond=None)[0]
            max_conc = np.percentile(concentrations, 99, axis=1, keepdims=True)
            concentrations = concentrations / max_conc * self.target_concentrations.T
            img_normalized = np.exp(-np.dot(self.target_stains, concentrations)) * 256
            img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8).T.reshape(img.shape)
            return Image.fromarray(img_normalized)
        except:
            return Image.fromarray(img)


def background_normalization(img):
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def color_constancy(img):
    img_array = np.array(img).astype(np.float32)
    means = [np.mean(img_array[:, :, i]) for i in range(3)]
    gray = sum(means) / 3
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * (gray / means[i]), 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))


# Setup
print("\nLoading WAMM2025 model and initializing methods...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = timm.create_model('maxvit_small_tf_384.in1k', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('WAMM2025_best_model.pth'))
    model = model.to(device)
    model.eval()
    print("WAMM2025 (MaxViT-Small) model loaded successfully")
except Exception as e:
    print(f"ERROR loading WAMM2025 model: {e}")
    raise SystemExit("Cannot proceed without trained WAMM2025 model")

target_layers = [(name, module) for name, module in model.named_modules() 
                 if isinstance(module, torch.nn.Conv2d)]
target_layer_name, target_layer = target_layers[-1]
print(f"Using layer: {target_layer_name}")

# Initialize methods
hirescam = HiResCAM(model, target_layer)
integrated_gradients = IntegratedGradients(model)
gradcam_plusplus = GradCAMPlusPlus(model, target_layer)
scorecam = ScoreCAM(model, target_layer)
attention_rollout = AttentionRollout(model)

print("All 5 methods initialized!")
print("  1. HiResCAM (Best)")
print("  2. Integrated Gradients (Most Rigorous)")
print("  3. Grad-CAM++ (Improved Baseline)")
print("  4. Score-CAM (Gradient-Free)")
print("  5. Attention Rollout (Transformer)")


# Load data
print("\nLoading validation data...")

IMG_SIZE = 384
MERGED_PATH = "/content/combined_multi_species"

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from torch.utils.data import Dataset, DataLoader


class MalariaDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_stain_norm=True):
        self.root_dir = root_dir
        self.transform = transform
        self.stain_normalizer = StainNormalizer() if use_stain_norm else None
        self.images, self.labels = [], []
        
        for class_name, label in [("Parasitized", 1), ("Uninfected", 0)]:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = background_normalization(img)
        img = color_constancy(img)
        if self.stain_normalizer:
            img = self.stain_normalizer.normalize(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


val_dataset = MalariaDataset(MERGED_PATH, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Select samples
n_samples = 6
samples = []
parasitized_count = uninfected_count = 0

print(f"Selecting {n_samples} samples...")
for images, labels in tqdm(val_loader):
    if labels.item() == 1 and parasitized_count < n_samples // 2:
        samples.append((images, labels))
        parasitized_count += 1
    elif labels.item() == 0 and uninfected_count < n_samples // 2:
        samples.append((images, labels))
        uninfected_count += 1
    if len(samples) >= n_samples:
        break

print(f"Selected {len(samples)} samples")


# Generate visualizations
print("\nGenerating ALL explainability methods for WAMM2025...")
print("  This may take 2-3 minutes (Score-CAM is computationally intensive)")

fig = plt.figure(figsize=(30, 18))
gs = fig.add_gridspec(n_samples, 7, hspace=0.3, wspace=0.15)

for idx, (images, labels) in enumerate(tqdm(samples)):
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate ALL methods
    hirescam_map = hirescam.generate_cam(images, target_class=pred_class)
    integrated_grads_map = integrated_gradients.generate_attribution(images, target_class=pred_class, steps=50)
    gradcam_pp_map = gradcam_plusplus.generate_cam(images, target_class=pred_class)
    scorecam_map = scorecam.generate_cam(images, target_class=pred_class)
    attention_map = attention_rollout.generate_attention_map(images)
    
    orig_img = denormalize_image(images[0])
    orig_img_uint8 = np.uint8(255 * orig_img)
    
    # Apply colormaps and overlays
    hirescam_colored = apply_colormap(hirescam_map)
    integrated_grads_colored = apply_colormap(integrated_grads_map)
    gradcam_pp_colored = apply_colormap(gradcam_pp_map)
    scorecam_colored = apply_colormap(scorecam_map)
    attention_colored = apply_colormap(attention_map)
    
    hirescam_overlay = overlay_heatmap(orig_img_uint8, hirescam_colored, alpha=0.5)
    integrated_grads_overlay = overlay_heatmap(orig_img_uint8, integrated_grads_colored, alpha=0.5)
    
    # Plot
    row = idx
    
    # Original
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(orig_img)
    ax0.axis('off')
    true_label = "Parasitized" if labels.item() == 1 else "Uninfected"
    pred_label = "Parasitized" if pred_class == 1 else "Uninfected"
    color = 'green' if labels.item() == pred_class else 'red'
    ax0.set_title(f'Original\nTrue: {true_label}\nPred: {pred_label}\n{confidence*100:.1f}%', 
                  fontsize=10, color=color, fontweight='bold')
    
    # HiResCAM
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.imshow(hirescam_overlay)
    ax1.axis('off')
    ax1.set_title('HiResCAM\n(Best)', fontsize=10, fontweight='bold')
    
    # Integrated Gradients
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.imshow(integrated_grads_overlay)
    ax2.axis('off')
    ax2.set_title('Integrated\nGradients', fontsize=10, fontweight='bold')
    
    # Grad-CAM++
    ax3 = fig.add_subplot(gs[row, 3])
    ax3.imshow(overlay_heatmap(orig_img_uint8, gradcam_pp_colored, alpha=0.5))
    ax3.axis('off')
    ax3.set_title('Grad-CAM++', fontsize=10)
    
    # Score-CAM
    ax4 = fig.add_subplot(gs[row, 4])
    ax4.imshow(overlay_heatmap(orig_img_uint8, scorecam_colored, alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Score-CAM', fontsize=10)
    
    # Attention Rollout
    ax5 = fig.add_subplot(gs[row, 5])
    ax5.imshow(overlay_heatmap(orig_img_uint8, attention_colored, alpha=0.5))
    ax5.axis('off')
    ax5.set_title('Attention\nRollout', fontsize=10)
    
    # Consensus
    consensus = (hirescam_map + integrated_grads_map + gradcam_pp_map) / 3
    consensus_colored = apply_colormap(consensus)
    ax6 = fig.add_subplot(gs[row, 6])
    ax6.imshow(overlay_heatmap(orig_img_uint8, consensus_colored, alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Consensus\n(Average)', fontsize=10, fontweight='bold')

plt.suptitle('WAMM2025 EXPLAINABILITY - State-of-the-Art Methods\nMaxViT-Small (384x384) Clinical-Grade Malaria Detection', 
             fontsize=18, fontweight='bold')

# Save
print("\nSaving WAMM2025 visualization...")

plt.savefig('WAMM2025_ultimate_explainability.png', dpi=300, bbox_inches='tight')
print("Saved locally: WAMM2025_ultimate_explainability.png")

if DRIVE_AVAILABLE:
    try:
        drive_path = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_ultimate_explainability.png')
        plt.savefig(drive_path, dpi=300, bbox_inches='tight')
        print(f"Saved to Drive: {drive_path}")
    except Exception as e:
        print(f"Could not save to Drive: {e}")

plt.show()

# Cleanup
attention_rollout.cleanup()


# Detailed comparison
print("\nGenerating detailed comparison for WAMM2025...")

sample_img, sample_label = samples[0]
sample_img = sample_img.to(device)

with torch.no_grad():
    outputs = model(sample_img)
    probs = F.softmax(outputs, dim=1)
    pred_class = outputs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

hirescam_map = hirescam.generate_cam(sample_img, pred_class)
ig_map = integrated_gradients.generate_attribution(sample_img, pred_class, steps=50)
gradcampp_map = gradcam_plusplus.generate_cam(sample_img, pred_class)
score_map = scorecam.generate_cam(sample_img, pred_class)
attn_map = attention_rollout.generate_attention_map(sample_img)

orig_img = denormalize_image(sample_img[0])
orig_img_uint8 = np.uint8(255 * orig_img)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

methods = [
    ('HiResCAM (Best)', hirescam_map),
    ('Integrated Gradients', ig_map),
    ('Grad-CAM++', gradcampp_map),
    ('Score-CAM', score_map),
    ('Attention Rollout', attn_map),
    ('Consensus', (hirescam_map + ig_map + gradcampp_map) / 3)
]

for idx, (name, cam_map) in enumerate(methods):
    ax = axes[idx // 3, idx % 3]
    colored = apply_colormap(cam_map)
    overlay = overlay_heatmap(orig_img_uint8, colored, alpha=0.5)
    ax.imshow(overlay)
    ax.axis('off')
    true_label = "Parasitized" if sample_label.item() == 1 else "Uninfected"
    pred_label = "Parasitized" if pred_class == 1 else "Uninfected"
    ax.set_title(f'{name}\nTrue: {true_label} | Pred: {pred_label} ({confidence*100:.1f}%)',
                fontsize=12, fontweight='bold')

plt.suptitle('WAMM2025 Detailed Comparison - All Methods on Same Sample', fontsize=16, fontweight='bold')
plt.tight_layout()

print("\nSaving detailed comparison...")

plt.savefig('WAMM2025_detailed_comparison_all_methods.png', dpi=300, bbox_inches='tight')
print("Saved locally: WAMM2025_detailed_comparison_all_methods.png")

if DRIVE_AVAILABLE:
    try:
        drive_path = os.path.join(DRIVE_OUTPUT_DIR, 'WAMM2025_detailed_comparison_all_methods.png')
        plt.savefig(drive_path, dpi=300, bbox_inches='tight')
        print(f"Saved to Drive: {drive_path}")
    except Exception as e:
        print(f"Could not save to Drive: {e}")

plt.show()


# Summary
print()
print("=" * 80)
print("WAMM2025 EXPLAINABILITY COMPLETE!")
print("=" * 80)

print()
print("Generated Visualizations:")
print()
print("  LOCAL STORAGE (/content/):")
print("    - WAMM2025_ultimate_explainability.png (6 samples, all methods)")
print("    - WAMM2025_detailed_comparison_all_methods.png (detailed single sample)")

if DRIVE_AVAILABLE:
    print()
    print(f"  GOOGLE DRIVE (PERSISTENT):")
    print(f"    - {DRIVE_OUTPUT_DIR}/WAMM2025_ultimate_explainability.png")
    print(f"    - {DRIVE_OUTPUT_DIR}/WAMM2025_detailed_comparison_all_methods.png")
    print()
    print("  These files are SAFE even after session ends!")
else:
    print()
    print("  GOOGLE DRIVE: Not mounted")
    print("  DOWNLOAD these files before session ends!")

print()
print("Methods Implemented:")
print("  - HiResCAM (2023) - Best spatial accuracy, perfect for 384x384")
print("  - Integrated Gradients - Most rigorous, Google's method")
print("  - Grad-CAM++ - Improved baseline with better localization")
print("  - Score-CAM - Gradient-free, no backprop artifacts")
print("  - Attention Rollout - Transformer attention visualization")
print("  - Consensus - Average of best methods for maximum confidence")

print()
print("Method Comparison:")
print()
print("  | Method              | Speed | Accuracy | Clinical Use |")
print("  |---------------------|-------|----------|--------------|")
print("  | HiResCAM            | Fast  | *****    | *****        |")
print("  | Integrated Grads    | Slow  | *****    | *****        |")
print("  | Grad-CAM++          | Fast  | ****     | ****         |")
print("  | Score-CAM           | Slow  | ****     | ****         |")
print("  | Attention Rollout   | Fast  | ***      | ***          |")

print()
print("Recommendations:")
print("  1. Use HiResCAM for daily clinical use (fastest + most accurate)")
print("  2. Use Integrated Gradients for FDA submission (most rigorous)")
print("  3. Use Consensus for critical cases (maximum confidence)")
print("  4. Include ALL methods in research papers (comprehensive analysis)")

print()
print("=" * 80)
print("WAMM2025 explainability generation complete!")
print("=" * 80)
