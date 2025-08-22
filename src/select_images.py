import os
import random
import shutil

# --- Configuration ---

# 1. Define the source path where your images are located.
#    IMPORTANT: Replace 'user' with your actual Linux username.
source_directory = '/home/user/Desktop/Tsoutsas/myopia_project/data/HPMI/PM'

# 2. Define the destination path for the randomly selected images.
#    This will be created inside your 'data' folder.
destination_directory = os.path.join(os.path.dirname(source_directory), 'HPMI_random_700')

# 3. Set the number of images you want to select.
num_images_to_select = 700

# --- Main Script Logic ---

def select_random_images(src_dir, dest_dir, num_to_select):
    """
    Finds all images in a source directory, randomly selects a specified
    number of them, and copies them to a new destination directory.
    """
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dest_dir}")

    # Ensure the source directory exists
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found at '{src_dir}'")
        return

    # Create the destination directory, deleting it first if it already exists
    if os.path.exists(dest_dir):
        print(f"Removing existing destination directory: '{dest_dir}'")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    print(f"Created new destination directory: '{dest_dir}'")

    # Find all valid image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(valid_extensions)
    ]

    print(f"Found {len(all_images)} total images in the source directory.")

    # Check if there are enough images to select from
    if len(all_images) < num_to_select:
        print(f"Warning: Only {len(all_images)} images available, which is less than {num_to_select}.")
        print("Copying all available images instead.")
        images_to_copy = all_images
    else:
        # Randomly select the specified number of images
        images_to_copy = random.sample(all_images, num_to_select)
        print(f"Randomly selected {len(images_to_copy)} images.")

    # Copy the selected images to the destination directory
    print("Copying images...")
    for i, image_name in enumerate(images_to_copy):
        source_path = os.path.join(src_dir, image_name)
        destination_path = os.path.join(dest_dir, image_name)
        shutil.copy2(source_path, destination_path) # copy2 preserves metadata
        # Optional: print progress
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(images_to_copy)} images...")

    print(f"\nProcess complete. {len(images_to_copy)} images have been copied to '{dest_dir}'")

# --- Run the script ---
if __name__ == "__main__":
    # IMPORTANT: Make sure to update the 'source_directory' path above
    # with your correct username before running!
    select_random_images(source_directory, destination_directory, num_images_to_select)