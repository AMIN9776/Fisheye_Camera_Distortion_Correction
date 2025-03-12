import os
import random
import cv2 as cv
import numpy as np
from math import sqrt

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
downsamplex = downsampley = 1024

# input_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/train'
# output_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/distorted_train_1024/'
# original_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/rectified_train_1024/'


# input_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/val'
# output_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/distorted_val_1024/'
# original_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/rectified_val_1024/'


# input_folder = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/train'
# output_folder = './distorted_val/'
# original_folder = './rectified_val/'
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_fish_xn_yn(source_x, source_y, radius, coefficients):
    """
    Apply fisheye distortion using a 4-term polynomial model.
    rc = k1*r_d + k2*r_d^3 + k3*r_d^5 + k4*r_d^7
    """
    # Compute the distorted radius
    distorted_radius = sum(coefficients[i] * radius**(2 * i + 1) for i in range(len(coefficients)))

    # Avoid division by zero
    if radius == 0 or distorted_radius == 0:
        return source_x, source_y

    # Scale source coordinates based on distorted radius
    scale = distorted_radius / radius
    return source_x * scale, source_y * scale

def fish(img, coefficients):
    """
    Apply fisheye distortion to the input image using a 4-term polynomial model.
    """
    h, w = img.shape[:2]

    # Ensure the image has 3 channels
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    dstimg = np.zeros_like(img)

    for x in range(h):
        for y in range(w):
            # Normalize coordinates to [-1, 1]
            xnd, ynd = (2 * x - h) / h, (2 * y - w) / w
            rd = sqrt(xnd**2 + ynd**2)  # Radius in normalized coordinates

            # Skip the center point to avoid division by zero
            if rd == 0:
                continue

            # Apply distortion
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, coefficients)

            # Map back to pixel space
            xu, yu = int((xdu + 1) * h / 2), int((ydu + 1) * w / 2)

            if 0 <= xu < h and 0 <= yu < w:
                dstimg[x, y] = img[xu, yu]

    return dstimg

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(original_folder, exist_ok=True)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    try:
        # Generate random coefficients for the 4-term polynomial distortion model
        coefficients = [random.uniform(1, 2) for _ in range(4)]  # Larger range for noticeable distortion

        input_path = os.path.join(input_folder, image_file)
        img = cv.imread(input_path)

        if img is None:
            print(f"Failed to load image {input_path}")
            continue

        # Resize the original image to match the distorted output size
        resized_original = cv.resize(img, (downsamplex, downsampley))

        # Save the resized original image
        original_path = os.path.join(original_folder, f"original_{image_file}")
        cv.imwrite(original_path, resized_original)

        # Apply fisheye distortion
        distorted_img = fish(resized_original, coefficients)

        # Save the distorted image
        output_path = os.path.join(output_folder, f"training_fisheye_{image_file}")
        cv.imwrite(output_path, distorted_img)

        print(f"Processed {image_file} with coefficients {coefficients}")

    except Exception as e:
        print(f"An error occurred while processing {image_file}: {e}")
