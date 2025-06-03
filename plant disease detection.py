
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import os

# Path to the main directory containing the disease folders
data_dir = "D:\plant_images_dataset"  # Replace with your actual path
class_names = sorted(os.listdir(data_dir))  # List of folders (disease types)

# Loop through each folder (class) and read images
for class_name in class_names:
    folder_path = os.path.join(data_dir, class_name)  # Path to each class folder
    if os.path.isdir(folder_path):  # Check if it's a directory
        print(f"Reading images from folder: {class_name}")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Read and display image
            image = cv2.imread(file_path)
            if image is not None:
                print(f"Loaded image: {file_name}")
            else:
                print(f"Failed to load image: {file_name}")
import cv2
import os

# Path to the main directory containing the disease folders
data_dir = "D:/plant_images_dataset"  # Replace with your actual path
class_names = sorted(os.listdir(data_dir))  # List of folders (disease types)

# Loop through each folder (class) and count images
for class_name in class_names:
    folder_path = os.path.join(data_dir, class_name)  # Path to each class folder
    if os.path.isdir(folder_path):  # Check if it's a directory
        image_count = 0
        print(f"Reading images from folder: {class_name}")
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if file is an image (you can add more extensions if necessary)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image = cv2.imread(file_path)
                
                if image is not None:
                    image_count += 1  # Increment the count for each valid image
                else:
                    print(f"Failed to load image: {file_name}")
        
        # Print the count of images in the current folder
        print(f"Total images in {class_name}: {image_count}")
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np

# Path to the main directory containing the disease folders
data_dir = "D:/plant_images_dataset"  # Replace with your actual path
class_names = sorted(os.listdir(data_dir))  # List of folders (disease types)

# Lists to store image data and labels
X = []  # Images
y = []  # Labels

# Loop through each folder (class) and read images
for class_idx, class_name in enumerate(class_names):
    folder_path = os.path.join(data_dir, class_name)  # Path to each class folder
    if os.path.isdir(folder_path):  # Check if it's a directory
        print(f"Reading images from folder: {class_name}")
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if file is an image (you can add more extensions if necessary)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image = cv2.imread(file_path)
                
                if image is not None:
                    image = cv2.resize(image, (128, 128))  # Resize image to fixed size
                    X.append(image)  # Append image to X
                    y.append(class_idx)  # Append corresponding label (class index)
                else:
                    print(f"Failed to load image: {file_name}")

# Convert to numpy arrays
X = np.array(X, dtype='float32') / 255.0  # Normalize images to range [0, 1]
y = np.array(y)

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the size of the train and test sets
print(f"Training set size: {X_train.shape[0]} images")
print(f"Test set size: {X_test.shape[0]} images")
# Basic libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# TensorFlow and Keras libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For train-test split (if not already split)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


model_1 = models.Sequential()
model_1.add(Conv2D(32, (3, 3), padding="same", input_shape=(128, 128, 3), activation="relu"))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Flatten())
model_1.add(Dense(128, activation="relu"))
model_1.add(Dense(3, activation="softmax"))

model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# New model
model_2 = models.Sequential()
model_2.add(Conv2D(32, (3, 3), padding="same", input_shape=(128,128, 3), activation="relu"))
model_2.add(MaxPooling2D(pool_size=(3, 3)))
model_2.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Flatten())
model_2.add(Dense(8, activation="relu"))
model_2.add(Dense(3, activation="softmax"))

model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train Model 1
history_1 = model_1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Train Model 2
history_2 = model_2.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
# Plot training and validation accuracy for Model 1
plt.plot(history_1.history['accuracy'], label='model 1 train accuracy')
plt.plot(history_1.history['val_accuracy'], label='model 1 val accuracy')
plt.title('Model 1 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation accuracy for Model 2
plt.plot(history_2.history['accuracy'], label='model 2 train accuracy')
plt.plot(history_2.history['val_accuracy'], label='model 2 val accuracy')
plt.title('Model 2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the 3D outputs to 1D
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(128, activation='relu'))

# Output layer (softmax for multi-class classification)
model.add(layers.Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

# Print the model summary
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


ResNet50
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the pretrained ResNet50 base
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)

# Final model
resnet_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
resnet_model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Summary
resnet_model.summary()
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Convert labels to one-hot
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Input shape must match your resized image shape
input_tensor = Input(shape=(128, 128, 3))

# Load ResNet50 base (without top)
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
base_model.trainable = False  # freeze base

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

resnet_model = Model(inputs=base_model.input, outputs=output)

# Compile
resnet_model.compile(optimizer=Adam(),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Fit
history_resnet = resnet_model.fit(X_train, y_train_cat,
                                  epochs=30,
                                  batch_size=32,
                                  validation_data=(X_test, y_test_cat))
# Plot training & validation accuracy
plt.plot(history_resnet.history['accuracy'], label='Train Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

resnet_model.save("resnet_model.h5")