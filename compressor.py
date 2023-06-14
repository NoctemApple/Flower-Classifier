from PIL import Image
import os

# Set the directory where the images are located
data_dir = r'C:\Users\Joshua\Desktop\Sampaguita'

# Set the directory to save the compressed images
output_dir = r'C:\Users\Joshua\Desktop\AugSam'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the compression quality (0-100, lower value means higher compression)
compression_quality = 80

# Loop over the images in the directory
for image_file in os.listdir(data_dir):
    # Read the image
    image_path = os.path.join(data_dir, image_file)
    image = Image.open(image_path)

    # Compress and save the image
    output_path = os.path.join(output_dir, image_file)
    image.save(output_path, optimize=True, quality=compression_quality)

print("Image compression completed.")