import os
import pickle
import time
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier 

def classify_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    
    # Load and preprocess the selected image
    img = Image.open(file_path)
    img = img.resize((50, 50))
    img = np.array(img)
    img = img.flatten()
    img = img.reshape(1, -1)
    img_scaled = scaler.transform(img)
    
    # Make prediction using the trained model
    prediction = best_estimator.predict(img_scaled)
    
    # Update the predicted label on the GUI
    predicted_label_text.set(categories[prediction[0]])

print("Running...")

# prepare data
input_dir = r'C:\Users\Joshua\Desktop\Flowers'
categories = ['gumamela', 'sunflower', 'tulip', 'sampaguita', 'rose', 'chrysanthemum']

if not os.path.exists(input_dir):
    print(f"Directory '{input_dir}' does not exist.")
    exit()

print("Resizing...")

data = []
labels = []
total_files = 0

# Count the total number of files
for category_idx, category in enumerate(categories):
    total_files += len(os.listdir(os.path.join(input_dir, category)))

start_time = time.time()  # Start measuring the execution time

with tqdm(total=total_files, desc="Processing Images") as pbar:
    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(input_dir, category)):
            if file.endswith(('.jpg','.jpeg')):
                img_path = os.path.join(input_dir, category, file)
                img = imread(img_path)
                img = resize(img, (50, 50))
                data.append(img.flatten())
                labels.append(category_idx)
                pbar.update(1)
            else:
                print(f"Skipping file '{file}' as it is not a valid image file.")

data = np.asarray(data)
labels = np.asarray(labels)

end_time = time.time()  # Stop measuring the execution time
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

print("Splitting...")

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("Scaling the data...")

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Handling missing values...")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
x_train_scaled = imputer.fit_transform(x_train_scaled)
x_test_scaled = imputer.transform(x_test_scaled)

print("x_train_scaled shape:", x_train_scaled.shape)
print("y_train shape:", y_train.shape)

print("Classifying...")

# train classifier
classifier = RandomForestClassifier()

parameters = [
    {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]},  # Parameter values for the first combination
    {'n_estimators': [50, 100, 150], 'max_depth': [None, 3, 6]}  # Parameter values for the second combination
    # Add more parameter combinations as needed
]

start_time = time.time()  # Start measuring the execution time

with tqdm(total=len(parameters), desc="Grid Search") as pbar:
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train_scaled, y_train.astype(int))

    # Update the progress bar
    pbar.update(1)

end_time = time.time()  # Stop measuring the execution time
execution_time = end_time - start_time

print(f"Grid search execution time: {execution_time} seconds")
print("Testing Performance...")

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test_scaled)

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
