import random
import os
from PIL import Image


def random_crop(image, crop_size):
    width, height = image.size
    target_width, target_height = crop_size

    if width < target_width or height < target_height:
        raise ValueError("Image dimensions are smaller than the target crop size.")

    # Generate random crop positions
    left = random.randint(0, width - target_width)
    upper = random.randint(0, height - target_height)
    right = left + target_width
    lower = upper + target_height

    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image


def resize_image(image, target_size):
    resized_image = image.resize(target_size, Image.LANCZOS)
    return resized_image


# Specify the input directory containing the images
input_dir = r'C:\Users\Joshua\Desktop\test\tulip'

# Specify the output directory for the cropped and resized images
output_dir = input_dir  # Same as input directory

# Specify the crop size
crop_size = (30, 30)

# Specify the resize size
resize_size = (50, 50)

# Specify the number of crops
num_crops = 2

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the input directory
for image_file in os.listdir(input_dir):
    # Check if the file has a valid image extension
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)

        # Perform random cropping
        cropped_images = []
        for _ in range(num_crops):
            cropped_image = random_crop(image, crop_size)
            cropped_images.append(cropped_image)

        # Resize the cropped images
        resized_images = []
        for cropped_image in cropped_images:
            resized_image = resize_image(cropped_image, resize_size)
            resized_images.append(resized_image)

        # Save the original image with a new filename
        original_filename = os.path.splitext(image_file)[0]
        original_image_path = os.path.join(output_dir, f"{original_filename}_001.jpg")
        image.save(original_image_path)

        # Save the cropped and resized images with sequential filenames
        for i, resized_image in enumerate(resized_images, start=1):
            new_file_name = f"{original_filename}_{str(i+1).zfill(3)}.jpg"
            new_image_path = os.path.join(output_dir, new_file_name)
            resized_image.save(new_image_path)

        # Remove the original image if desired
        os.remove(image_path)

print("Images have been cropped, resized, and renamed successfully.")
