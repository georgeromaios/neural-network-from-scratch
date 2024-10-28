# mnist_digit_classifier.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Flatten 28x28 images into 784-element vectors
    Dense(128, activation='relu'),       # First hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax')      # Output layer with 10 neurons (one per digit) and softmax activation
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Step 6: Make predictions on the first 5 test images (optional visualization)
predictions = model.predict(x_test[:5])

# Plot the first 5 test images with predicted labels
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
