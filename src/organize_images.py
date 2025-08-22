import pandas as pd
import os
import shutil

# --- Please Edit This Line ---
# Change this to the folder where your images are located.
image_source_folder = '/home/user/Desktop/Tsoutsas/myopia_project/data/testing_data/ODIR-5K/ODIR-5K/Training Images'  #<-- IMPORTANT: SET THIS PATH!

# --- Script Starts Here ---

# Load the CSV data
try:
    df = pd.read_csv('full_df.csv')
except FileNotFoundError:
    print("Error: 'full_df.csv' not found. Make sure it's in the same folder as the script.")
    exit()


# Define the output folders
output_folder = 'organized_images'
class_folders = {
    'N': os.path.join(output_folder, 'Normal'),
    'H': os.path.join(output_folder, 'High Myopia'),
    'M': os.path.join(output_folder, 'Pathological Myopia')
}

# Create the folders if they don't exist
for folder in class_folders.values():
    os.makedirs(folder, exist_ok=True)

print("Starting to organize images...")

# Loop through the data and copy files
for index, row in df.iterrows():
    # The 'labels' column is a string like "['N']". We need to get the 'N' out.
    label = row['labels'].strip("[]'\"")

    if label in class_folders:
        # Get the filename and destination folder
        filename = row['filename']
        destination_folder = class_folders[label]

        # Create the full source and destination paths
        source_path = os.path.join(image_source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        # Copy the file, if it exists
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            print(f"Copied: {filename} -> {destination_folder}")
        else:
            print(f"Warning: Could not find {filename} in {image_source_folder}")

print("\nImage organization complete! organized_images folder is ready.")