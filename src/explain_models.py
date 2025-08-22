# =====================================================================================
# Myopia Classification – High-Accuracy Explainability with Dual-Rollout Attention
# -----------------------------------------------------------------------------
# This script generates comprehensive, high-sensitivity heatmaps by combining
# attention from both early (low-level) and late (high-level) transformer layers.
# =====================================================================================
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# --------------------------- 1. CONFIG (Must match your training script) ---------------------------
PROJECT_PATH = "."
DATA_PATH = os.path.join(PROJECT_PATH, "data")
REPORTS_PATH = os.path.join(PROJECT_PATH, "reports_experiment_REG1")
MODELS_PATH = os.path.join(PROJECT_PATH, "models_experiment_REG1")

SEED = 42
N_SPLITS = 10
IMAGE_SIZE = 518
MODEL_NAME = "vit_small_patch14_dinov2" # Has 12 transformer blocks (layers 0-11)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -------------------------- 2. REUSED COMPONENTS ---------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def prepare_dataframe(data_dir):
    class_folders = ["High_Myopia", "Normal", "Pathological_Myopia"]
    paths, labels = [], []
    for cls in class_folders:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    df = pd.DataFrame({"filepath": paths, "label": labels})
    label_map = {l: i for i, l in enumerate(sorted(df["label"].unique()))}
    df["label_idx"] = df["label"].map(label_map)
    return df, label_map

class ViTClassifier(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(self.backbone.num_features, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))

def get_base_transform(img_size):
    return Compose([Resize(img_size, img_size), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

# -------------------------- 3. DUAL-ROLLOUT ATTENTION (HIGH ACCURACY) ---------------------------------
def generate_comprehensive_attention_map(model, input_tensor):
    """
    Directly computes a comprehensive attention map by combining attention from
    both early and late layers to capture both fine details and high-level concepts.
    """
    # Manually perform the forward pass to get attentions from all layers
    attentions = []
    with torch.no_grad():
        x = model.backbone.patch_embed(input_tensor)
        x = torch.cat((model.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + model.backbone.pos_embed
        
        for blk in model.backbone.blocks:
            x_norm = blk.norm1(x)
            attn_matrix = blk.attn.get_attention_map(x_norm)
            attentions.append(attn_matrix.mean(dim=1)) # Average across heads
            x = blk(x)

    num_layers = len(attentions)
    grid_size = int((x.shape[1] - 1)**0.5)
    
    # --- Early Layer Rollout (for fine details) ---
    early_rollout = torch.eye(x.shape[1], device=DEVICE)
    early_layers = range(num_layers // 2) # First half of the layers
    for i in early_layers:
        attn_matrix = attentions[i]
        identity = torch.eye(attn_matrix.shape[-1], device=DEVICE)
        attn_matrix = attn_matrix + identity
        attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)
        early_rollout = torch.matmul(attn_matrix, early_rollout)
        
    early_map = early_rollout[0, 0, 1:].reshape(grid_size, grid_size)
    early_map = (early_map - early_map.min()) / (early_map.max() - early_map.min() + 1e-8)
    
    # --- Late Layer Rollout (for high-level concepts) ---
    late_rollout = torch.eye(x.shape[1], device=DEVICE)
    late_layers = range(num_layers // 2, num_layers) # Second half of the layers
    for i in late_layers:
        attn_matrix = attentions[i]
        identity = torch.eye(attn_matrix.shape[-1], device=DEVICE)
        attn_matrix = attn_matrix + identity
        attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)
        late_rollout = torch.matmul(attn_matrix, late_rollout)
        
    late_map = late_rollout[0, 0, 1:].reshape(grid_size, grid_size)
    late_map = (late_map - late_map.min()) / (late_map.max() - late_map.min() + 1e-8)
    
    # --- Combine Maps ---
    # Give slightly more weight to the high-level (late layer) map
    combined_map = 0.4 * early_map + 0.6 * late_map
    
    return combined_map.cpu().numpy()

def get_attention_map(self, x):
    qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    return attn.softmax(dim=-1)

# -------------------------- 4. PUBLICATION-QUALITY VISUALIZATION (DEFINITIVE) ---------------------------------
def save_explanation(image_path, heatmap, save_path, true_label, pred_label, original_size=(518, 518)):
    """
    Creates the definitive, publication-quality visualization.
    - Uses a circular mask to remove edge artifacts.
    - Uses an adaptive threshold to ensure full coverage of abnormalities.
    - Renders a vibrant green-to-red colormap that is not washed out.
    """
    original_img = Image.open(image_path).convert("RGB").resize(original_size)
    original_array = np.array(original_img)

    # --- HEATMAP & MASK PROCESSING ---

    # 1. Standard processing: normalize, upscale, and smooth the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap_upsampled = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_CUBIC)
    heatmap_smoothed = gaussian_filter(heatmap_upsampled, sigma=15)
    
    # 2. RE-INTRODUCE THE CIRCULAR MASK ("BLACK FILTER")
    # This removes any explanation glow from the black background corners.
    center_x, center_y = original_size[0] // 2, original_size[1] // 2
    radius = min(center_x, center_y) * 0.98
    Y, X = np.ogrid[:original_size[1], :original_size[0]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
    circular_mask = dist_from_center <= radius
    heatmap_smoothed *= circular_mask # Apply the mask
    
    # 3. Use the adaptive threshold for good coverage on large and small features
    max_val = np.max(heatmap_smoothed)
    if max_val > 0: # Avoid division by zero on empty maps
        heatmap_smoothed /= max_val # Explicitly normalize to 0-1 range after masking
        
    threshold = np.max(heatmap_smoothed) * 0.40
    concentrated_mask = (heatmap_smoothed > threshold).astype(float)
    concentrated_mask = gaussian_filter(concentrated_mask, sigma=15)
    
    # --- PROFESSIONAL COLORING & VIBRANT BLENDING ---
    
    # 4. Create a vibrant Green -> Yellow -> Red colormap
    colors = [(0, 0.6, 0), (1, 1, 0), (1, 0, 0)] # Dark Green -> Yellow -> Red
    custom_cmap = LinearSegmentedColormap.from_list('custom_gyr', colors, N=256)
    colored_heatmap = custom_cmap(heatmap_smoothed)[:, :, :3]
    
    # 5. Apply a more advanced blending formula to make colors pop
    # This formula makes the overlay more opaque where the mask is strong
    alpha = concentrated_mask[..., np.newaxis] * 0.9 # Master opacity
    
    # Blend the images
    overlay = (original_array / 255.0) * (1 - alpha) + (colored_heatmap * alpha)
    overlay = np.clip(overlay, 0, 1)

    # --- PLOTTING ---
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    title_color = 'lightgreen' if true_label == pred_label else 'salmon'
    fig.suptitle(f'True Label: {true_label}  |  Predicted Label: {pred_label}', 
                 color=title_color, fontsize=16, y=0.95)

    axes[0].imshow(original_array)
    axes[0].set_title('Original Image', color='white', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Explanation Overlay', color='white', fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close()

# -------------------------- 5. MAIN PIPELINE ---------------------------------
def main():
    EXPLAINABILITY_PATH = os.path.join(REPORTS_PATH, "explainability_high_accuracy")
    os.makedirs(EXPLAINABILITY_PATH, exist_ok=True)

    df, label_map = prepare_dataframe(DATA_PATH)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    _, test_df = train_test_split(df, test_size=0.15, stratify=df.label_idx, random_state=SEED)

    N_EXPLAIN_IMGS = 5
    explain_df = test_df.sample(N_EXPLAIN_IMGS, random_state=SEED)
    print(f"🔬 Generating High-Accuracy Attention maps for {N_EXPLAIN_IMGS} images per fold...")
    print(f"💾 Results will be saved to: {EXPLAINABILITY_PATH}")

    base_tf = get_base_transform(IMAGE_SIZE)

    timm.models.vision_transformer.Attention.get_attention_map = get_attention_map

    for fold in range(N_SPLITS):
        print(f"\n--- Processing Fold {fold + 1}/{N_SPLITS} ---")
        model = ViTClassifier(MODEL_NAME, num_classes=len(label_map)).to(DEVICE)
        model_path = os.path.join(MODELS_PATH, f"fold{fold+1}_best.pth")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        for row in tqdm(list(explain_df.itertuples()), desc=f"Fold {fold+1} Final Heatmaps"):
            img_pil = Image.open(row.filepath).convert("RGB")
            img_tensor = base_tf(image=np.array(img_pil))["image"].unsqueeze(0).to(DEVICE)
            
            true_class = class_names[row.label_idx]
            with torch.no_grad():
                pred_class = class_names[model(img_tensor).argmax().item()]
            
            heatmap = generate_comprehensive_attention_map(model, img_tensor)

            save_filename = f"final_heatmap_fold_{fold+1}_img_{row.Index}_true_{true_class}_pred_{pred_class}.png"
            save_path = os.path.join(EXPLAINABILITY_PATH, save_filename)
            
            save_explanation(row.filepath, heatmap, save_path, true_label=true_class, pred_label=pred_class)

    print("\n✅ High-accuracy explainability analysis complete.")

if __name__ == "__main__":
    main()