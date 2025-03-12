import cv2
import numpy as np
import os
import glob
from scipy.ndimage import distance_transform_edt

def fill_black_border_deep(image, threshold=0, erode_iterations=5, kernel_size=3):
    """
    Fills the black border of an image by taking the color value from a pixel that is 
    not only non-black, but also slightly deeper inside the image.
    
    Parameters:
      image           : Input image as a NumPy array (BGR format assumed).
      threshold       : A pixel is considered non-black if any channel > threshold.
      erode_iterations: Number of iterations for erosion (controls how deep the source pixel is).
      kernel_size     : Size of the kernel used for erosion.
    
    Returns:
      filled          : The image with the black border padded from the deeper non-black pixels.
    """
    # Step 1: Build a mask where pixels are considered valid if any channel is above the threshold.
    mask = np.any(image > threshold, axis=2)  # True for non-black pixels

    if np.all(~mask):
        # If the entire image is black, nothing to do.
        return image

    # Step 2: Erode the mask to get an inner region.
    # This erosion “pushes” the valid region inward, ensuring that the source pixel is not too close to the border.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=erode_iterations).astype(bool)

    # If the erosion removes the entire region (which can happen if erode_iterations is too high),
    # fall back to the original mask.
    if not np.any(inner_mask):
        inner_mask = mask

    # Step 3: Compute a distance transform on the inverse of the inner_mask.
    # For every pixel (including those in the border) we find the nearest pixel that lies in inner_mask.
    distances, indices = distance_transform_edt(~inner_mask, return_indices=True)

    # Create a copy of the original image for output.
    filled = image.copy()

    # Identify pixels that are not in the inner region (i.e. border pixels).
    border_pixels = np.where(~inner_mask)
    # Replace each border pixel with the value from the nearest inner pixel.
    filled[border_pixels] = image[indices[0][border_pixels], indices[1][border_pixels]]

    return filled

def process_directory_deep(input_dir, output_dir, threshold=0, erode_iterations=5, kernel_size=3):
    """
    Processes all images in the input directory. For each image the function
    fills in the black curved border using pixels from deeper inside the image.
    The processed images are saved to the output directory (with the same filename).
    
    Parameters:
      input_dir       : Directory containing input images.
      output_dir      : Directory where processed images will be saved.
      threshold       : Threshold to consider a pixel as non-black.
      erode_iterations: How many iterations to erode the mask (controls depth).
      kernel_size     : Kernel size used for the erosion.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Grab all files (adjust the pattern if you want only specific image types)
    image_paths = glob.glob(os.path.join(input_dir, '*'))

    for path in image_paths:
        # Read the image.
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {path} (not a valid image?)")
            continue

        # Process the image to fill the black border.
        new_img = fill_black_border_deep(img, threshold=threshold,
                                         erode_iterations=erode_iterations,
                                         kernel_size=kernel_size)

        # Save the processed image.
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, new_img)
        print(f"Processed and saved: {out_path}")


if __name__ == '__main__':
    # Replace these with the paths to your directories.
    # input_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_train_1024'
    # output_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_pad_train_1024'
    
    input_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_val_1024'
    output_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_pad_val_1024'
    
    # If you want to treat near-black as black, you can set threshold to a small value (e.g. 10)
    process_directory_deep(input_directory, output_directory,
                           threshold=3, erode_iterations=6, kernel_size=3)
