import os
import cv2

# Set the directory where the original images are located
data_dir = r'C:\Users\Joshua\Desktop\augros'

# Set the directory to save the renamed images
output_dir = r'C:\Users\Joshua\Desktop\reros'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the directory
for image_file in os.listdir(data_dir):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Read the image
        image_path = os.path.join(data_dir, image_file)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, (50, 50))

        # Save the resized image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, resized_image)

    else:
        print(f"Skipping file: {image_file} - Invalid image format")
