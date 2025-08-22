import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations import (
    CLAHE,
    Compose,
    GaussNoise,
    GridDropout,
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
    ShiftScaleRotate,
)

# --- Configuration ---
# Create a directory to save the reports
REPORTS_PATH = "augmentation_reports"
os.makedirs(REPORTS_PATH, exist_ok=True)

# Paths for both left and right eye images
IMG_PATHS = [
    "/home/user/Desktop/Tsoutsas/myopia_project/data/normal/1_left.jpg",
    "/home/user/Desktop/Tsoutsas/myopia_project/data/normal/1_right.jpg",
]
IMAGE_SIZE = 224

# --- Define Augmentations from the Original Script ---
augmentations = {
    "HorizontalFlip": HorizontalFlip(p=1.0),
    "ShiftScaleRotate": ShiftScaleRotate(p=1.0),
    "RandomBrightnessContrast": RandomBrightnessContrast(p=1.0),
    "CLAHE": CLAHE(p=1.0),
    "GaussNoise": GaussNoise(p=1.0),
    "RandomGamma": RandomGamma(p=1.0),
    "GridDropout": GridDropout(ratio=0.5, p=1.0),
}

# --- Main Loop ---
# Loop through each image (left and right eye)
for image_path in IMG_PATHS:
    print(f"--- Processing image: {image_path} ---")
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except (FileNotFoundError, cv2.error):
        print(f"⚠️ Error: Could not read image at '{image_path}'. Skipping.")
        continue

    base_filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    
    # Set a consistent random seed for reproducibility
    np.random.seed(42)

    # Loop through each augmentation to create a separate plot for it
    for name, aug in augmentations.items():
        # --- Create a new plot for each augmentation ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Technique: {name}\non {os.path.basename(image_path)}', fontsize=16)

        # Apply the specific augmentation
        pipeline = Compose([Resize(IMAGE_SIZE, IMAGE_SIZE), aug])
        augmented_image = pipeline(image=original_image)['image']

        # --- Plot Original Image ---
        ax = axes[0]
        resized_original = cv2.resize(original_image, (IMAGE_SIZE, IMAGE_SIZE))
        ax.imshow(resized_original)
        ax.set_title("Original")
        ax.axis('off')

        # --- Plot Augmented Image ---
        ax = axes[1]
        ax.imshow(augmented_image)
        ax.set_title(f"Augmented: {name}")
        ax.axis('off')

        # --- Save the individual plot ---
        output_filename = f"aug_{base_filename_no_ext}_{name}.png"
        save_path = os.path.join(REPORTS_PATH, output_filename)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory before the next loop
        
        print(f"✅ Saved plot: {output_filename}")

print("\nAll processing complete.")