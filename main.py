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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Loading...")

def classify_image():
    # Open file dialog to choose an image
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if file_path:
        # Load and preprocess the image
        img = imread(file_path)
        img = resize(img, (15, 15))
        img_flattened = img.flatten()

        # Classify the image using the best estimator
        prediction = best_estimator.predict([img_flattened])[0]

        # Update the prediction label in the GUI
        predicted_label_text.set(f"Predicted Label: {categories[prediction]}")

        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo

# prepare data
input_dir = r'C:\Users\Joshua\Desktop\Flowers'
categories = ["chrysanthemum", "gumamela", "rose", "sampaguita", "sunflower", "tulip"]

if not os.path.exists(input_dir):
    print(f"Directory '{input_dir}' does not exist.")
    exit()

print("Resizing...")

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in tqdm(os.listdir(os.path.join(input_dir, category))):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

print("Splitting...")

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("Classifying...")

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))

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
