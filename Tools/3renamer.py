import os
import shutil

# Set the directory where the original images are located
data_dir = r'C:\Users\Joshua\Desktop\retul'

# Set the directory to save the renamed images
output_dir = r'C:\Users\Joshua\Desktop\tulip'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the directory
for i, image_file in enumerate(sorted(os.listdir(data_dir))):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Generate the new file name
        new_filename = f'rose_{str(i+1).zfill(3)}.jpg'

        # Rename the image file
        image_path = os.path.join(data_dir, image_file)
        new_image_path = os.path.join(output_dir, new_filename)
        os.rename(image_path, new_image_path)

print("File names have been renamed successfully.")