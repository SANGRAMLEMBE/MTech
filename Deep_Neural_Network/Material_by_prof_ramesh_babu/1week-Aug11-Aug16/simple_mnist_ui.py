#!/usr/bin/env python3

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw, ImageTk

# Create and train a simple MNIST model
def create_mnist_model():
    """Create and train a simple MNIST MLP model"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to flatten the images
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training model...")
    # Train model (just a few epochs for demo)
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1, validation_split=0.1)
    
    # Save model
    model.save("mnist_mlp.keras")
    print("Model saved as mnist_mlp.keras")
    
    return model

# Load or create model
try:
    model = tf.keras.models.load_model("mnist_mlp.keras")
    print("Loaded existing model")
except:
    print("No existing model found, creating new one...")
    model = create_mnist_model()

def preprocess_img(img_array):
    """
    Preprocess the drawn image for MNIST prediction
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Invert colors (make background black, digit white)
    gray = 255 - gray
    
    # Find bounding box
    ys, xs = np.where(gray > 50)  # threshold for non-zero pixels
    if len(xs) == 0 or len(ys) == 0:
        # Empty drawing
        roi = np.zeros((28, 28), dtype=np.uint8)
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = gray[y1:y2+1, x1:x2+1]
        
        # Make square by padding
        h, w = crop.shape
        s = max(h, w)
        pad_y = (s - h) // 2
        pad_x = (s - w) // 2
        crop_sq = cv2.copyMakeBorder(crop, pad_y, s - h - pad_y, pad_x, s - w - pad_x,
                                     cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 28x28
        roi = cv2.resize(crop_sq, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize and reshape
    roi = roi.astype('float32') / 255.0
    roi = roi.reshape(1, 28*28)
    return roi

class MNISTDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition")
        self.root.geometry("600x500")
        
        # Create canvas for drawing
        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        # Image for drawing
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        clear_btn = Button(button_frame, text="Clear", command=self.clear_canvas, font=("Arial", 12))
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        predict_btn = Button(button_frame, text="Predict", command=self.predict, font=("Arial", 12))
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Result label
        self.result_label = Label(root, text="Draw a digit and click Predict", font=("Arial", 14))
        self.result_label.pack(pady=20)
        
        # Confidence labels
        self.confidence_label = Label(root, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=10)
        
    def paint(self, event):
        # Draw on canvas
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        
        # Draw on PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
        
    def predict(self):
        # Convert PIL image to numpy array
        img_array = np.array(self.image)
        
        # Preprocess
        roi = preprocess_img(img_array)
        
        if roi is None:
            self.result_label.config(text="No drawing detected")
            return
            
        # Predict
        probs = model.predict(roi, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        
        # Show result
        self.result_label.config(text=f"Predicted: {pred}", fg="blue")
        
        # Show top 3 predictions
        top3_idx = probs.argsort()[-3:][::-1]
        top3_text = "Top 3: "
        for i, idx in enumerate(top3_idx):
            top3_text += f"{idx}({probs[idx]:.2f}) "
        self.confidence_label.config(text=top3_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawingApp(root)
    root.mainloop()
