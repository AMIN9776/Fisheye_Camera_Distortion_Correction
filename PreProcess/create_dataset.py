import os
import random
import shutil
from math import ceil, floor
#------------------------------------------------------------------------------------------------------

#main_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Places365/archive/train'  # Path to the main folder containing subfolders
#destination_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/train'  # Destination folder for the dataset
#N = 20000  # Total number of images
#M = 365   # Number of subfolders to pick from




main_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Places365/archive/val'  # Path to the main folder containing subfolders
destination_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/val'  # Destination folder for the dataset
N = 1000 # Total number of images
M = 365    # Number of subfolders to pick from

#------------------------------------------------------------------------------------------------------
import os
import random
import shutil
from math import ceil, floor

def build_dataset(main_folder, destination_folder, N, M, seed=42):
    """
    Build a dataset by selecting N images evenly distributed across M subfolders.
    Ensures the same images are picked every time by using a fixed random seed.

    Parameters:
        main_folder (str): Path to the main folder containing subfolders with images.
        destination_folder (str): Path to the folder where the selected images will be saved.
        N (int): Total number of images to select.
        M (int): Number of subfolders to select from.
        seed (int): Random seed for reproducibility.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Get all subfolders in the main folder
    subfolders = [os.path.join(main_folder, folder) for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]
    num_subfolders = len(subfolders)
    
    if num_subfolders == 0:
        raise ValueError("No subfolders found in the main folder.")
    
    if num_subfolders > M:
        print(f"Only {num_subfolders} subfolders available, reducing M to {num_subfolders}.")
        M = num_subfolders  # Adjust M if fewer subfolders are available

    # Randomly select M subfolders
    selected_subfolders = random.sample(subfolders, M)

    # Calculate the number of images to pick from each subfolder
    images_per_folder = N / M
    lower = floor(images_per_folder)
    upper = ceil(images_per_folder)
    total_images = 0

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for i, folder in enumerate(selected_subfolders):
        # Adjust the number of images to maintain the total of N
        if i < (N % M):  # Distribute remainders evenly
            num_images = upper
        else:
            num_images = lower

        # Get all images in the subfolder
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < num_images:
            raise ValueError(f"Subfolder {folder} has only {len(images)} images, but {num_images} were requested.")
        
        # Randomly select images from the subfolder (fixed order due to seed)
        selected_images = random.sample(images, num_images)
        folder_name = os.path.basename(folder)
        # Copy selected images to the destination folder
            
            # Copy with new name
            
        
        
        for img_path in selected_images:
            original_name = os.path.basename(img_path)
            new_name = f"{os.path.splitext(original_name)[0]}_{folder_name}{os.path.splitext(original_name)[1]}"
            shutil.copy(img_path, os.path.join(destination_folder, new_name))
            #shutil.copy(img_path, destination_folder)
        
        total_images += num_images
        print(f"Selected {num_images} images from {folder}.")

    print(f"Dataset built with {total_images} images in {destination_folder}.")


build_dataset(main_folder, destination_folder, N, M, seed=42)
