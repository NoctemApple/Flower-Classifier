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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import uniform

# Global variable to store the scaled image
img_scaled = None

def classify_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()

    if file_path:
        # Load and preprocess the selected image
        with open(file_path, 'rb') as fp:
            img = Image.open(fp)
            img = img.resize((50, 50))
            img = np.array(img)
            img = img.flatten()
            img = img.reshape(1, -1)
            img_scaled = scaler.transform(img)  # Update the global variable

        # Make prediction using the trained model
        prediction = best_estimator.predict(img_scaled)

        # Update the predicted label on the GUI
        predicted_label_text.set(categories[prediction[0]])

        # Display the selected image
        img_tk = ImageTk.PhotoImage(Image.open(file_path).resize((200, 200)))
        image_label.configure(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
    else:
        print("No file selected.")


print("Running...")

# prepare data
input_dir = r'C:\Users\Joshua\Desktop\Flowers'
categories = ["chrysanthemum", "gumamela", "rose", "sampaguita", "sunflower", "tulip"]

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
            if file.endswith(('.jpg', '.jpeg')):
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

start_time = time.time()  # Start measuring the execution time

# Define the classifier and its hyperparameter distribution
classifier = RandomForestClassifier()
param_dist = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10)

# Fit the random search on your data with progress bar
try:
    with tqdm(total=10) as pbar:  # Set the total number of iterations for the progress bar
        def update_pbar(estimator, x, y):
            pbar.update(1)  # Update the progress bar after each iteration

        random_search.fit(x_train_scaled, y_train)

except KeyboardInterrupt:
    print("Training process interrupted by user.")



# Print the best hyperparameters found
print("Best Hyperparameters: ", random_search.best_params_)

end_time = time.time()  # Stop measuring the execution time
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

print("Testing Performance...")

# Test performance
best_estimator = random_search.best_estimator_

# Perform testing using the best estimator
y_prediction = best_estimator.predict(x_test_scaled)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))

print("Creating GUI...")

# Load the model
best_estimator = pickle.load(open('./model.p', 'rb'))

# Verify if the model is loaded correctly
if best_estimator is not None:
    print("Model loaded successfully")
else:
    print("Error loading the model")
    
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
