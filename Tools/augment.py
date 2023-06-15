import os
import cv2
import numpy as np

print("Augmenting beep beep...")

# Set the directory where the original images are located
data_dir = r'C:\Users\Joshua\Desktop\auggum'

# Set the directory to save the augmented images
output_dir = r'C:\Users\Joshua\Desktop\fligum'

# Define the rotation angles
rotation_angles = [90, 180, 270]

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the directory
for image_file in os.listdir(data_dir):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Read the image
        image = cv2.imread(os.path.join(data_dir, image_file))

        # # Apply rotation for each angle
        # for angle in rotation_angles:
        #     # Rotate the image
        #     rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_180)

        #     # Save the rotated image with angle suffix
        #     angle_suffix = str(angle) + 'deg'
        #     output_filename = 'rotated_' + angle_suffix + '_' + image_file
        #     output_path = os.path.join(output_dir, output_filename)
        #     cv2.imwrite(output_path, rotated_image)

        #   # Apply flipping operations
        # flipped_image = cv2.flip(image, 1)  # Horizontal flip
        # flipped_image2 = cv2.flip(image, 0)  # Vertical flip
        # flipped_image3 = cv2.flip(image, -1)  # Both horizontal and vertical flip

        # # Save the flipped images
        # cv2.imwrite(os.path.join(output_dir, 'flipped_' + image_file), flipped_image)
        # cv2.imwrite(os.path.join(output_dir, 'flipped2_' + image_file), flipped_image2)
        # cv2.imwrite(os.path.join(output_dir, 'flipped3_' + image_file), flipped_image3)

        # Add Gaussian noise to the images
        # mean = 0
        # stddev = 20  # Adjust the standard deviation to control the amount of noise
        # noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        # noisy_image = cv2.add(image, noise)

        # # Save the augmented images
        # cv2.imwrite(os.path.join(output_dir, 'noisy_' + image_file), noisy_image)

            
    else:
        print(f"Skipping file: {image_file} - Invalid image format")