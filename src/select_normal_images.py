import os
import pandas as pd
import random
import shutil

# --- Configuration ---
# NOTE: No changes are needed here unless you move your folders.
base_path = '/home/user/Desktop/Tsoutsas/myopia_project/data/1. OIA-ODIR/On-site Test Set'
SOURCE_IMAGES_DIR = os.path.join(base_path, 'Images')
ANNOTATION_FILE_PATH = os.path.join(base_path, 'Annotation', 'on-site test annotation (English).xlsx')
DESTINATION_DIR = '/home/user/Desktop/Tsoutsas/myopia_project/data/Normal_random_700'
NUM_TO_SELECT = 700

# The keyword we are looking for in the diagnosis columns
NORMAL_LABEL_VALUE = 'normal'

# --- Main Script Logic ---

def select_and_copy_images(src_dir, dest_dir, annotation_path, num_to_select):
    """
    Reads the Excel file, finds all 'normal' images for both left and right eyes,
    and copies a random selection of them to a new directory.
    """
    print("--- Starting Process ---")
    print(f"Reading annotations from: {annotation_path}")

    try:
        df = pd.read_excel(annotation_path)
        # Clean up column names to remove any potential extra spaces
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: Annotation file not found at '{annotation_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return

    # Create a list to hold the filenames of all normal images
    normal_filenames = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check the left eye
        if NORMAL_LABEL_VALUE in str(row['Left-Diagnostic Keywords']).lower():
            normal_filenames.append(row['Left-Fundus'])
        
        # Check the right eye
        if NORMAL_LABEL_VALUE in str(row['Right-Diagnostic Keywords']).lower():
            normal_filenames.append(row['Right-Fundus'])

    if not normal_filenames:
        print(f"Error: No images containing the label '{NORMAL_LABEL_VALUE}' were found.")
        return

    print(f"Found {len(normal_filenames)} total normal images for both left and right eyes.")

    # Select the random images
    if len(normal_filenames) < num_to_select:
        print(f"Warning: Only {len(normal_filenames)} normal images available, which is less than {num_to_select}.")
        print("Selecting all available normal images instead.")
        selected_filenames = normal_filenames
    else:
        selected_filenames = random.sample(normal_filenames, num_to_select)
        print(f"Randomly selected {len(selected_filenames)} normal image filenames.")

    # Create destination directory
    if os.path.exists(dest_dir):
        print(f"Removing existing destination directory: '{dest_dir}'")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    print(f"Created new destination directory: '{dest_dir}'")

    # Copy the selected files
    print("Copying images...")
    copied_count = 0
    not_found_count = 0
    for filename in selected_filenames:
        if pd.isna(filename):  # Skip if the filename is empty
            continue
            
        source_path = os.path.join(src_dir, filename)
        destination_path = os.path.join(dest_dir, filename)

        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        else:
            print(f"  - Warning: Image '{filename}' not found in source directory. Skipping.")
            not_found_count += 1

    print("\n--- Process Complete ---")
    print(f"Successfully copied {copied_count} images to '{dest_dir}'.")
    if not_found_count > 0:
        print(f"Could not find {not_found_count} images in the source directory.")

# --- Run the script ---
if __name__ == "__main__":
    select_and_copy_images(SOURCE_IMAGES_DIR, DESTINATION_DIR, ANNOTATION_FILE_PATH, NUM_TO_SELECT)