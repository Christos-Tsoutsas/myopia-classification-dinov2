# =====================================================================================
# Myopia Classification – EfficientNet-B3 Experiment
# -----------------------------------------------------------------------------
# Modern CNN baseline using EfficientNet-B3:
#   • ImageNet-pre-trained EfficientNet-B3 backbone.
#   • Standard input resolution (300²) for EfficientNet-B3.
#   • Differential learning-rates when unfreezing (backbone << head).
#   • Focal-Loss with label smoothing.
#   • All logging / plotting utilities retained.
#
# Author: Christos Tsoutsas – University of Thessaly, Greece
# Date: July 2025
# =====================================================================================

import os
import random
import warnings
from datetime import datetime
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- 1. CONFIG ---------------------------------------------
PROJECT_PATH = "."
DATA_PATH = os.path.join(PROJECT_PATH, "data")
REPORTS_PATH = os.path.join(PROJECT_PATH, "reports_experiment_efficientnet_b3")
MODELS_PATH = os.path.join(PROJECT_PATH, "models_experiment_efficientnet_b3")

SEED = 42
N_SPLITS = 10
HEAD_TUNE_EPOCHS = 10        # stage-1
FULL_TUNE_EPOCHS = 40        # stage-2

BATCH_SIZE = 16              # Reduced for larger images
IMAGE_SIZE = 300             # Standard EfficientNet-B3 input resolution
HEAD_LR = 2e-4
FULL_TUNE_LR = 2e-6          # for backbone (head will get ×10)
WEIGHT_DECAY = 0.05
MODEL_NAME = "tf_efficientnet_b3.ns_jft_in1k"  # An effective EfficientNet-B3 pretrained on ImageNet-1k

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. REPRODUCIBILITY -------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------- 3. DATA PREPARATION ------------------------------------

def prepare_dataframe(data_dir):
    class_folders = ["High_Myopia", "Normal", "Pathological_Myopia"]
    paths, labels = [], []
    for cls in class_folders:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    if not paths:
        raise RuntimeError("No images found in DATA_PATH")
    df = pd.DataFrame({"filepath": paths, "label": labels})
    label_map = {l: i for i, l in enumerate(sorted(df["label"].unique()))}
    df["label_idx"] = df["label"].map(label_map)
    return df, label_map

# -------------------------- 4. AUGMENTATIONS ---------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size):
    train_tf = Compose([
        Resize(img_size, img_size),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = Compose([
        Resize(img_size, img_size),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf

# -------------------------- 5. DATASET --------------------------------------------


class MyopiaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row.filepath), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(row.label_idx, dtype=torch.long)


# -------------------------- 6. MODEL ----------------------------------------------


class MyopiaClassifier(nn.Module):
    """
    CNN/ViT backbone + fresh linear classification head.
    """
    def __init__(self, model_name: str, num_classes: int, pretrained=True, in_chans=3, img_size=224):
        super().__init__()
        # Create the backbone with the original head removed
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # --- FIX: Determine the feature size dynamically ---
        # Create a dummy input tensor to pass through the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, in_chans, img_size, img_size)
            n_features = self.backbone(dummy_input).shape[1]
        
        # Now create the head with the correctly determined number of features
        self.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# -------------------------- 7. LOSS FUNCTION --------------------------------------


class FocalWithLS(nn.Module):
    """Focal-loss modulated cross-entropy with label smoothing"""

    def __init__(self, gamma: float = 2.0, ls: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = ls

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, label_smoothing=self.ls, reduction="none")
        p_t = torch.exp(-ce)  # same as probability of the true class
        return ((1 - p_t) ** self.gamma * ce).mean()


# -------------------------- 8. TRAIN / VAL LOOPS ----------------------------------


def train_one_epoch(model, loader, crit, optim, scaler, sched):
    model.train()
    epoch_loss, correct, total, step = 0.0, 0, 0, 0
    for imgs, lbls in tqdm(loader, desc="Train", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optim.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = crit(logits, lbls)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        sched.step_update(num_updates=step)
        step += 1
        epoch_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == lbls).sum().item()
        total += lbls.size(0)
    return epoch_loss / total, correct / total


def validate(model, loader, crit):
    model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with autocast():
                logits = model(imgs)
                loss = crit(logits, lbls)
            epoch_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            y_true.extend(lbls.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return epoch_loss / total, acc, macro_f1


# -------------------------- 9. MAIN PIPELINE --------------------------------------


def main():
    os.makedirs(REPORTS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    print(f"[INFO] {datetime.now():%Y-%m-%d %H:%M} – Starting EfficientNet-B3 Experiment on {DEVICE}")

    df, label_map = prepare_dataframe(DATA_PATH)

    # Leave the plotting / histogram code unchanged from previous experiments if desired

    trainval_df, test_df = train_test_split(df, test_size=0.15, stratify=df.label_idx, random_state=SEED)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_fold_val_acc = {}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(trainval_df, trainval_df.label_idx)):
        print(f"\n{'=' * 18} FOLD {fold + 1}/{N_SPLITS} {'=' * 18}")
        tr_df = trainval_df.iloc[tr_idx]
        val_df = trainval_df.iloc[val_idx]

        train_tf, val_tf = get_transforms(IMAGE_SIZE)
        train_ds = MyopiaDataset(tr_df, transform=train_tf)
        val_ds = MyopiaDataset(val_df, transform=val_tf)

        # --- class-balanced sampler as before -------------------------
        class_counts = tr_df.label_idx.value_counts().sort_index().values
        inv_freq = 1.0 / class_counts
        weights = torch.DoubleTensor([inv_freq[i] for i in tr_df.label_idx])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = MyopiaClassifier(
            MODEL_NAME,
            num_classes=len(label_map),
            img_size=IMAGE_SIZE
        ).to(DEVICE)
        criterion = FocalWithLS(gamma=2.0, ls=0.05)
        scaler = GradScaler()
        best_f1 = -1
        history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}

        # ---------- STAGE 1 – train head --------------------------------
        for p in model.backbone.parameters():
            p.requires_grad = False
        optim_head = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        sched_head = CosineLRScheduler(optim_head, t_initial=HEAD_TUNE_EPOCHS * len(train_loader), lr_min=1e-6,
                                       warmup_t=2 * len(train_loader), warmup_lr_init=1e-6)

        print("➡️  Stage 1: training classification head")
        for epoch in range(HEAD_TUNE_EPOCHS):
            tl, ta = train_one_epoch(model, train_loader, criterion, optim_head, scaler, sched_head)
            vl, va, vf1 = validate(model, val_loader, criterion)
            history["train_loss"].append(tl); history["train_acc"].append(ta)
            history["val_loss"].append(vl);   history["val_acc"].append(va)
            print(f"E{epoch + 1:02}/{HEAD_TUNE_EPOCHS} – TL:{tl:.4f} TA:{ta:.3f} | VL:{vl:.4f} VA:{va:.3f} F1:{vf1:.3f}")

        # ---------- STAGE 2 – fine-tune all layers ----------------------
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": FULL_TUNE_LR},
            {"params": model.head.parameters(),     "lr": FULL_TUNE_LR * 10},
        ], weight_decay=WEIGHT_DECAY)
        scheduler = CosineLRScheduler(optimizer, t_initial=FULL_TUNE_EPOCHS * len(train_loader), lr_min=1e-7,
                                      warmup_t=5 * len(train_loader), warmup_lr_init=1e-7)

        print("➡️  Stage 2: fine-tuning entire EfficientNet-B3")
        for epoch in range(FULL_TUNE_EPOCHS):
            tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            vl, va, vf1 = validate(model, val_loader, criterion)
            history["train_loss"].append(tl); history["train_acc"].append(ta)
            history["val_loss"].append(vl);   history["val_acc"].append(va)
            print(f"E{epoch + 1:02}/{FULL_TUNE_EPOCHS} – TL:{tl:.4f} TA:{ta:.3f} | VL:{vl:.4f} VA:{va:.3f} F1:{vf1:.3f}")

            if vf1 > best_f1:
                best_f1 = vf1
                ckpt = os.path.join(MODELS_PATH, f"fold{fold + 1}_best.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"  💾 New best model saved (macro-F1={best_f1:.3f})")

        # =====================================================================
        #       PLOT TRAINING HISTORY FOR THE CURRENT FOLD (WITH METRICS BOX)
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Fold {fold + 1} Training Curves', fontsize=16)

        # --- Best Metrics Calculation ---
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])
        metrics_text = (f"Best Val Loss: {best_val_loss:.4f}\n"
                        f"Best Val Acc:  {best_val_acc:.4f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Plot Loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot Accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Val Accuracy')
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Add metrics box to the second plot
        ax2.text(0.95, 0.05, metrics_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)

        # Save the figure
        plot_path = os.path.join(REPORTS_PATH, f"training_curves_fold_{fold + 1}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"  📈 Fold history plot with metrics saved to {plot_path}")

        # store val accuracy curve
        all_fold_val_acc[f"Fold {fold + 1}"] = history["val_acc"]

    # =====================================================================
    #    PLOT COMBINED & AVERAGE VALIDATION ACCURACY (WITH METRICS BOX)
    # =====================================================================
    all_acc_df = pd.DataFrame(all_fold_val_acc)
    all_acc_df['average_validation_accuracy'] = all_acc_df.mean(axis=1)

    # --- Best Metric Calculation ---
    best_avg_acc = all_acc_df['average_validation_accuracy'].max()
    metrics_text = f"Peak Avg. Val Accuracy: {best_avg_acc:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # --- Plotting ---
    plt.figure(figsize=(14, 8))
    ax = plt.gca() # Get current axes

    for col in all_acc_df.columns:
        if col.startswith("Fold"):
            plt.plot(all_acc_df.index, all_acc_df[col], lw=1.5, alpha=0.4)
    plt.plot(all_acc_df.index, all_acc_df['average_validation_accuracy'], 'k-', lw=2.5, label='Average Validation Accuracy')

    plt.title('Validation Accuracy Across All Folds - EfficientNet-B3 Experiment')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--')

    # Add metrics box
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, "all_folds_validation_accuracy.png"))
    plt.close()
    print("📊 Combined fold accuracy plot with metrics saved.")

    # ----------------------------- FINAL TEST ----------------------------------
    print("\n######## FINAL TEST ########")
    _, val_tf = get_transforms(IMAGE_SIZE)
    test_ds = MyopiaDataset(test_df, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Ensemble predictions from all folds
    all_fold_preds = []
    for fold in range(N_SPLITS):
        model = MyopiaClassifier(
            MODEL_NAME,
            num_classes=len(label_map),
            img_size=IMAGE_SIZE
        ).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, f"fold{fold+1}_best.pth"), weights_only=True))
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in tqdm(test_loader, desc=f"Fold {fold+1}"):
                with autocast():
                    logits = model(imgs.to(DEVICE))
                preds.append(F.softmax(logits, 1).cpu())
        all_fold_preds.append(torch.cat(preds))

    # Average predictions across folds
    mean_preds = torch.stack(all_fold_preds).mean(0)
    y_pred = mean_preds.argmax(1).numpy()
    y_true = test_df.label_idx.values
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # =====================================================================
    #       GENERATE ALL FINAL EVALUATION PLOTS (WITH METRICS)
    # =====================================================================
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    y_score = mean_preds.numpy()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ## ------ 1. Classification Report Heatmap ------
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T
    macro_f1 = report_dict['macro avg']['f1-score'] # Get best overall metric
    metrics_text = f"Macro Avg. F1-Score: {macro_f1:.4f}"

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    ax.set_title('Classification Report Heatmap - EfficientNet-B3 Experiment', pad=40)
    ax.text(0.5, 1.08, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    plt.savefig(os.path.join(REPORTS_PATH, "classification_report_heatmap.png"))
    plt.close()

    ## ------ 2. Normalized Confusion Matrix ------
    # This plot's primary metrics are the diagonal values (recalls),
    # so an extra box is redundant. It remains unchanged.
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix - EfficientNet-B3 Experiment')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(REPORTS_PATH, "normalized_confusion_matrix.png"))
    plt.close()

    ## ------ 3. ROC Curves and AUC Score ------
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    best_auc = max(roc_auc.values())
    metrics_text = f"Best Class AUC: {best_auc:.4f}"

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC of {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves - EfficientNet-B3 Experiment')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(REPORTS_PATH, "roc_curves_and_auc.png"))
    plt.close()

    ## ------ 4. Precision-Recall (PR) Curves ------
    precision, recall, avg_precision = dict(), dict(), dict()
    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
    best_ap = max(avg_precision.values())
    metrics_text = f"Best Class Avg. Precision: {best_ap:.4f}"

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR for {class_names[i]} (AP = {avg_precision[i]:.2f})')
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curves - EfficientNet-B3 Experiment'); plt.legend(loc="best")
    plt.savefig(os.path.join(REPORTS_PATH, "precision_recall_curves.png"))
    plt.close()

    # Final classification report
    rep_str = classification_report(y_true, y_pred, target_names=class_names)
    print(rep_str)
    with open(os.path.join(REPORTS_PATH, "final_report.txt"), "w") as f:
        f.write(rep_str)

    # Standard Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - EfficientNet-B3 Experiment")
    plt.savefig(os.path.join(REPORTS_PATH, "confusion_matrix.png"))
    plt.close()

    print("\n✅ All requested plots with metrics have been generated and saved.")
    print(f"[INFO] EfficientNet-B3 Experiment completed. Results saved to {REPORTS_PATH}")


if __name__ == "__main__":
    main()
