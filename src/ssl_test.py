from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np, torch, glob
import os
import pandas as pd
import cv2
import timm
import torch.nn as nn
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# Constants from SSL_clean.py
PROJECT_PATH = "."
DATA_PATH = os.path.join(PROJECT_PATH, "data")
SEED = 42
IMAGE_SIZE = 518
MODEL_NAME = "vit_base_patch14_dinov2"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class from SSL_clean.py
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

# Model class from SSL_clean.py
class ViTClassifier(nn.Module):
    """ViT backbone (DINO-v2) + linear classification head"""

    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        super().__init__()
        # num_classes=0 removes the original head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

# Data preparation function from SSL_clean.py
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

# Transform function from SSL_clean.py
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

# Setup data
df, label_map = prepare_dataframe(DATA_PATH)
trainval_df, test_df = train_test_split(df, test_size=0.15, stratify=df.label_idx, random_state=SEED)
_, val_tf = get_transforms(IMAGE_SIZE)
y_true = test_df.label_idx.values

test_ds = MyopiaDataset(test_df, transform=val_tf)
loader  = DataLoader(test_ds, batch_size=32)

acc, f1 = [], []
for ckpt in glob.glob('models_experiment_7/fold*_best.pth'):
    m = ViTClassifier(MODEL_NAME, len(label_map)).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True)); m.eval()
    y_pred = []
    with torch.no_grad():
        for x,_ in loader:
            y_pred.append(m(x.to(device)).argmax(1).cpu())
    y_pred = torch.cat(y_pred).numpy()
    acc.append(accuracy_score(y_true, y_pred))
    f1.append( f1_score   (y_true, y_pred, average='macro') )

print('Accuracy  mean±sd:', np.mean(acc), np.std(acc))
print('Macro-F1  mean±sd:', np.mean(f1), np.std(f1))