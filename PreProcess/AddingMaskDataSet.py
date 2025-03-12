import cv2
import numpy as np
import os

def create_advanced_mask_from_color_fisheye(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("The image file could not be loaded.")
    grayscale_image = np.max(image, axis=2)
    _, initial_mask = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return initial_mask
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(grayscale_image)
    cv2.fillPoly(mask, [largest_contour], 255)
    return mask

def apply_mask_to_rectified(image_path, mask):
    rectified_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rectified_image is None:
        raise FileNotFoundError("Failed to load the rectified image.")
    mask_expanded = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rectified_with_border = cv2.bitwise_and(rectified_image, mask_expanded)
    return rectified_with_border

def process_dataset(fisheye_directory, rectified_directory, output_directory):
    # List and sort all fisheye and rectified images
    fisheye_files = sorted(os.listdir(fisheye_directory))
    rectified_files = sorted(os.listdir(rectified_directory))

    # Make sure both directories have the same number of files
    if len(fisheye_files) != len(rectified_files):
        raise ValueError("The number of fisheye and rectified images does not match.")

    # Process each pair of sorted files
    for fisheye_filename, rectified_filename in zip(fisheye_files, rectified_files):
        fisheye_path = os.path.join(fisheye_directory, fisheye_filename)
        rectified_path = os.path.join(rectified_directory, rectified_filename)
        output_path = os.path.join(output_directory, "modified_" + rectified_filename)

        # Create mask and apply to rectified image
        mask = create_advanced_mask_from_color_fisheye(fisheye_path)
        rectified_with_border = apply_mask_to_rectified(rectified_path, mask)

        # Save the modified rectified image
        cv2.imwrite(output_path, rectified_with_border)
        print(f"Processed and saved: {output_path}")

# Specify directories
fisheye_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/crop_distorted_val'
rectified_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/rectified_val'
output_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/mask_rectified_val'

# Make sure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process the entire dataset
process_dataset(fisheye_directory, rectified_directory, output_directory)
