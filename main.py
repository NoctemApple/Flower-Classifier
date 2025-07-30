import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16
import tensorflow as tf

print("TF Version:", tf.__version__)

MODEL_PATH = "model.h5"
categories = ["chrysanthemum", "gumamela", "rose", "sampaguita", "sunflower", "tulip"]
input_dir = './Flowers'

def check_model():
    if os.path.exists(MODEL_PATH):
        root = tk.Tk()
        root.withdraw()
        answer = messagebox.askyesno(
            "Model Found",
            "A trained model already exists.\nDo you want to create a new model?"
        )
        return answer
    else:
        return True

should_create_new = check_model()

if should_create_new:
    print("Preparing data...")

    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist.")
        exit()

    data, labels = [], []
    for category_idx, category in enumerate(categories):
        for file in tqdm(os.listdir(os.path.join(input_dir, category))):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            img = resize(img, (64, 64))
            data.append(img)
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
        data = np.repeat(data, 3, axis=-1)

    print("Oversampling...")
    oversampler = RandomOverSampler()
    data_resampled, labels_resampled = oversampler.fit_resample(data.reshape(len(data), -1), labels)
    data_resampled = data_resampled.reshape(len(data_resampled), 64, 64, 3)

    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, stratify=labels_resampled)

    print("Building model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(x_train, y_train, batch_size=64, epochs=12, verbose=1)

    _, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_accuracy)

    model.save(MODEL_PATH)
    print("Model saved.")
else:
    print("Loading existing model...")
    model = load_model(MODEL_PATH)

# GUI setup
print("Creating GUI...")

window = tk.Tk()
window.title("Flower Classifier")
window.geometry("400x500")

predicted_label_text = tk.StringVar()
predicted_label_widget = tk.Label(window, textvariable=predicted_label_text)
predicted_label_widget.pack()

image_label = tk.Label(window)
image_label.pack()

def classify_image():
    window.lift()
    window.update()

    file_path = filedialog.askopenfilename(
        parent=window,
        title="Select an image",
        filetypes=[("Image files", "*.png"), ("Image files", "*.jpg"), ("Image files", "*.jpeg"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        image = Image.open(file_path).convert("RGB")
        image = image.resize((64, 64))
        tk_image = ImageTk.PhotoImage(image)

        image_label.configure(image=tk_image)
        image_label.image = tk_image  # 🔐 Must keep reference
        window.tk_image = tk_image    # 🔐 Additional safety for macOS

        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = categories[predicted_index]
        predicted_label_text.set(f"Prediction: {predicted_label}")

    except Exception as e:
        print(f"Error loading image: {e}")
        predicted_label_text.set("Error loading image.")



select_image_button = tk.Button(window, text="Select Image", command=classify_image)
select_image_button.pack()

window.after(500, lambda: None)
window.mainloop()
