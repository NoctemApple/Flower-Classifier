import os
import pickle
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score

print("Loading...")

def classify_image():
    # Open file dialog to choose an image
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if file_path:
        # Load and preprocess the image
        img = imread(file_path)
        img_resized = resize(img, (15, 15, 3))
        img_reshaped = np.expand_dims(img_resized, axis=0)  # Reshape the image for CNN input

        # Classify the image using the CNN model
        prediction = np.argmax(model.predict(img_reshaped))

        # Update the prediction label in the GUI
        predicted_label_text.set(f"Predicted Label: {categories[prediction]}")

        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo

# Prepare data
input_dir = r'C:\Users\Joshua\Desktop\Flowers'
categories = ["chrysanthemum", "gumamela", "rose", "sampaguita", "sunflower", "tulip"]

if not os.path.exists(input_dir):
    print(f"Directory '{input_dir}' does not exist.")
    exit()

print("Loading images...")

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in tqdm(os.listdir(os.path.join(input_dir, category))):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img_resized = resize(img, (15, 15, 3))
        if img_resized.shape == (15, 15):
            data.append(img_resized)  # Append the resized image directly
            labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

print("Splitting...")

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("Classifying...")

# CNN model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(15, 15, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 25
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# Evaluate the model
_, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

pickle.dump(model, open('./model.p', 'wb'))

print("Creating GUI...")

# Create a Tkinter window
window = tk.Tk()
window.title("Sample Predictions")
window.geometry("400x500")

# Create button to select an image
select_image_button = tk.Button(window, text="Select Image", command=classify_image)
select_image_button.pack()

# Create label for predicted label
predicted_label_text = tk.StringVar()
predicted_label_widget = tk.Label(window, textvariable=predicted_label_text)
predicted_label_widget.pack()

# Create label for displaying the image
image_label = tk.Label(window)
image_label.pack()

# Run the Tkinter event loop
window.mainloop()
