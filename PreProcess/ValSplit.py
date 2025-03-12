
import os
import random
import shutil




def main():
    # ================================
    # Set your paths and parameters here:
    # ================================
    
    # Paths for the source directories containing the paired images.
    source_dir1 = "/home/amin/Projects/Amin/Fisheye/PreProcess/crop_distorted_pad_val"  # e.g., original images
    source_dir2 = "/home/amin/Projects/Amin/Fisheye/PreProcess/rectified_val"  # e.g., corresponding labels
    
    # Paths for the target directories where the selected images will be copied.
    target_dir1 = "/home/amin/Projects/Amin/Fisheye/PreProcess/crop_distorted_pad_train"
    target_dir2 = "/home/amin/Projects/Amin/Fisheye/PreProcess/rectified_train"
    

    # Second pair of target directories for the m pairs.
    target_dir3 = "/home/amin/Projects/Amin/Fisheye/PreProcess/crop_distorted_pad_val1"  # e.g., second set: images from camera A
    target_dir4 = "/home/amin/Projects/Amin/Fisheye/PreProcess/rectified_val1"  # e.g., second set: images from camera B

    # Number of pairs to select:
    n = 4000  # Number of pairs for the first target directories
    m = 500# Number of pairs for the second target directories

    # =====================================================
    # End of configuration section.
    # =====================================================

    # Create target directories if they don't exist.
    for directory in [target_dir1, target_dir2, target_dir3, target_dir4]:
        os.makedirs(directory, exist_ok=True)

    # List and sort the files in both source directories.
    # Sorting ensures that pairing by order is maintained.
    files1 = sorted(os.listdir(source_dir1))
    files2 = sorted(os.listdir(source_dir2))
    
    if len(files1) != len(files2):
        print("Error: The two source directories do not contain the same number of files.")
        return
    
    total_files = len(files1)
    total_needed = n + m
    if total_files < total_needed:
        print(f"Error: Not enough images. Found {total_files} images but need {total_needed} pairs.")
        return
    
    # -----------------------------
    # Randomly select disjoint indices
    # -----------------------------
    # Create a set of all valid indices.
    all_indices = set(range(total_files))
    
    # Randomly choose n indices for the first group.
    selected_n = sorted(random.sample(all_indices, n))
    
    # Remove these indices from the pool.
    remaining_indices = list(all_indices - set(selected_n))
    
    # Randomly choose m indices from the remaining indices for the second group.
    selected_m = sorted(random.sample(remaining_indices, m))
    
    print("Selected indices for first group (n pairs):", selected_n)
    print("Selected indices for second group (m pairs):", selected_m)
    
    # -----------------------------
    # Copy pairs for the first target directories.
    # -----------------------------
    for idx in selected_n:
        file1 = files1[idx]
        file2 = files2[idx]
        
        src_path1 = os.path.join(source_dir1, file1)
        src_path2 = os.path.join(source_dir2, file2)
        
        dst_path1 = os.path.join(target_dir1, file1)
        dst_path2 = os.path.join(target_dir2, file2)
        
        shutil.copy2(src_path1, dst_path1)
        shutil.copy2(src_path2, dst_path2)
        print(f"First group: Copied pair (index {idx}): '{file1}' and '{file2}'")
    
    # -----------------------------
    # Copy pairs for the second target directories.
    # -----------------------------
    for idx in selected_m:
        file1 = files1[idx]
        file2 = files2[idx]
        
        src_path1 = os.path.join(source_dir1, file1)
        src_path2 = os.path.join(source_dir2, file2)
        
        dst_path1 = os.path.join(target_dir3, file1)
        dst_path2 = os.path.join(target_dir4, file2)
        
        shutil.copy2(src_path1, dst_path1)
        shutil.copy2(src_path2, dst_path2)
        print(f"Second group: Copied pair (index {idx}): '{file1}' and '{file2}'")

if __name__ == "__main__":
    main()
