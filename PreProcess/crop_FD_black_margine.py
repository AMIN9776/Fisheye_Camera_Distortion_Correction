import os
import cv2
import numpy as np

def crop_and_resize_images(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of image names with specific extensions in the input directory
    valid_extensions = ['.jpg', '.png', '.jpeg']
    image_names = [f for f in sorted(os.listdir(input_directory)) if f.lower().endswith(tuple(valid_extensions))]

    # Iterate through sorted images in the input directory
    for image_name in image_names:
        input_image_path = os.path.join(input_directory, image_name)

        # Read the RGB image
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        # Convert RGB to grayscale for finding non-zero pixels
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the first non-zero pixel position in each direction
        non_zero_indices = np.nonzero(gray_img)
        min_row, min_col = np.min(non_zero_indices[0]), np.min(non_zero_indices[1])
        max_row, max_col = np.max(non_zero_indices[0]), np.max(non_zero_indices[1])

        # Crop the black margin and resize the image to the original size
        cropped_img = img[min_row:max_row, min_col:max_col, :]
        resized_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))

        # Save the cropped and resized image as RGB with the new name to the output directory
        output_image_path = os.path.join(output_directory, f'C{image_name}')
        cv2.imwrite(output_image_path, resized_img)

if __name__ == "__main__":
    # Replace these paths with your actual directories
    # input_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/distorted_train_1024'
    # output_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_train_1024'
    
    input_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/distorted_val_1024'
    output_directory = '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_val_1024'

    # Call the function to crop and resize images
    crop_and_resize_images(input_directory, output_directory)
