import os
import shutil
import cv2

# Set the directory where the original images are located
data_dir = r'C:\Users\Joshua\Desktop\chr'

# Set the directory to save the separated RGB channels
output_dir = r'C:\Users\Joshua\Desktop\rgbchr'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the directory
for i, image_file in enumerate(sorted(os.listdir(data_dir))):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Read the image
        image_path = os.path.join(data_dir, image_file)
        image = cv2.imread(image_path)

        # Split the image into RGB channels
        red_channel = image[:, :, 2]  # Extract the red channel
        green_channel = image[:, :, 1]  # Extract the green channel
        blue_channel = image[:, :, 0]  # Extract the blue channel

        # Save the RGB channels as individual images
        red_channel_path = os.path.join(output_dir, f'{image_file.split(".")[0]}_red.jpg')
        green_channel_path = os.path.join(output_dir, f'{image_file.split(".")[0]}_green.jpg')
        blue_channel_path = os.path.join(output_dir, f'{image_file.split(".")[0]}_blue.jpg')

        cv2.imwrite(red_channel_path, red_channel)
        cv2.imwrite(green_channel_path, green_channel)
        cv2.imwrite(blue_channel_path, blue_channel)

print("RGB channels have been separated and saved successfully.")
