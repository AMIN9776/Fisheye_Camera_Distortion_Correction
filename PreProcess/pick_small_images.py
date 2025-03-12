import os
import shutil
import random

def select_random_images_with_seed(images1, num_images, seed):
    random.seed(seed)
    selected_indices = random.sample(range(len(images1)), num_images)
    return selected_indices

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def copy_paired_images(source_dir1, source_dir2, dest_dir1, dest_dir2, num_images):
    
    # Clear destination directories
    clear_directory(dest_dir1)
    clear_directory(dest_dir2)
   
    if not os.path.exists(dest_dir1):
        os.makedirs(dest_dir1)
    if not os.path.exists(dest_dir2):
        os.makedirs(dest_dir2)

    # Get list of images from both source directories
    images1 = sorted(os.listdir(source_dir1))
    images2 = sorted(os.listdir(source_dir2))

    # Check if the number of images to copy is valid
    num_images = min(num_images, len(images1), len(images2))
    if num_images <= 0:
        raise ValueError("num_images should be a positive integer and less than or equal to the number of images in the source directories")
    # Randomly select indices
    seed = 42

    selected_indices = select_random_images_with_seed(images1, num_images, seed)
    print(selected_indices)

    # Copy paired images
    for i in selected_indices:
        src_image1 = os.path.join(source_dir1, images1[i])
        src_image2 = os.path.join(source_dir2, images2[i])

        dest_image1 = os.path.join(dest_dir1, images1[i])
        dest_image2 = os.path.join(dest_dir2, images2[i])

        shutil.copy(src_image1, dest_image1)
        shutil.copy(src_image2, dest_image2)

    #print(f"Copied {num_images} paired images to {dest_dir1} and {dest_dir2}")

# Example usage


source_dir1 = '/home/ai/Projects/Amin/Fisheye/training_fisheye'
source_dir2 = '/home/ai/Projects/Amin/Fisheye/training_rectified'
dest_dir1 = '/home/ai/Projects/Amin/Fisheye/training_fisheye_small'
dest_dir2 = '/home/ai/Projects/Amin/Fisheye/training_rectified_small'



num_images = 2000  # Number of paired images to copy

copy_paired_images(source_dir1, source_dir2, dest_dir1, dest_dir2, num_images)
