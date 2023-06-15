from PIL import Image
import os

print("compressing...")

# Set the directory where the images are located
data_dir = r'C:\Users\Joshua\Desktop\fligum'

# Set the directory to save the compressed images
output_dir = r'C:\Users\Joshua\Desktop\comgum'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the compression quality (0-100, lower value means higher compression)
compression_quality = 80

# Loop over the images in the directory
for image_file in os.listdir(data_dir):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Read the image
        image_path = os.path.join(data_dir, image_file)
        image = Image.open(image_path)

        # Convert the image to 'RGB' mode
        image = image.convert('RGB')

        # Compress and save the image
        output_path = os.path.join(output_dir, image_file)
        image.save(output_path, optimize=True, quality=compression_quality)
    else:
        print(f"Skipping file: {image_file} - Invalid image format")

print("Image compression completed.")
