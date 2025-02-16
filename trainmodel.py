

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Path for the images
path = 'gray_images'

# Ensure a directory exists for saving the trained model
if not os.path.exists('./recognizer'):
    os.makedirs('./recognizer')  # Directory for saving the CNN model

# Function to load images and labels
def get_images_with_id(path):
    faces = []
    face_ids = []

    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            img_path = os.path.join(root, filename)
            label = os.path.basename(root)  # Folder name is used as the label
            print(f'Loading: {img_path} with label: {label}')

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            if img is None:
                print(f"Image {img_path} not loaded properly.")
                continue

            # Resize all images to 100x100 for consistency
            img_resized = cv2.resize(img, (100, 100))

            faces.append(img_resized)
            face_ids.append(int(label))

    return np.array(faces), np.array(face_ids)

# Load the dataset
faces, face_ids = get_images_with_id(path)

# Normalize pixel values to range [0, 1] and reshape for CNN input
faces = faces / 255.0  # Normalize
faces = faces.reshape(faces.shape[0], 100, 100, 1)  # Add channel dimension

print(f"Total images loaded: {faces.shape[0]}, Total classes: {len(np.unique(face_ids))}")

# Convert labels to categorical format for training
face_ids = to_categorical(face_ids)


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, face_ids, test_size=0.2, random_state=42)

# Create a data augmentation generator
data_aug = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(face_ids.shape[1], activation='softmax')  # Number of output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN with data augmentation
print("Training the CNN model...")
history = model.fit(
    data_aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=15
)

# Save the trained model
model.save('./recognizer/face_recognizer_cnn.h5')
print("Training completed and model saved.")

#Pretrained model

'''
import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Path for the images
path = 'gray_images'

# Ensure a directory exists for saving the trained model
if not os.path.exists('./recognizer'):
    os.makedirs('./recognizer')  # Directory for saving the model

# Function to load images and labels
def get_images_with_id(path):
    faces = []
    face_ids = []

    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            img_path = os.path.join(root, filename)
            label = os.path.basename(root)  # Folder name is used as the label
            print(f'Loading: {img_path} with label: {label}')

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            if img is None:
                print(f"Image {img_path} not loaded properly.")
                continue

            # Resize all images to 224x224 to match ResNet50 input
            img_resized = cv2.resize(img, (224, 224))

            faces.append(img_resized)
            face_ids.append(int(label))

    return np.array(faces), np.array(face_ids)

# Load the dataset
faces, face_ids = get_images_with_id(path)

# Normalize pixel values and preprocess for ResNet50
faces = preprocess_input(faces.astype('float32'))  # ResNet50 preprocessing
faces = faces.reshape(faces.shape[0], 224, 224, 1).repeat(3, axis=-1)  # Convert to 3 channels

print(f"Total images loaded: {faces.shape[0]}, Total classes: {len(np.unique(face_ids))}")

# Convert labels to categorical format
face_ids = to_categorical(face_ids)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, face_ids, test_size=0.2, random_state=42)

# Create a data augmentation generator
data_aug = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers for the specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(face_ids.shape[1], activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
print("Training the ResNet50 model...")
history = model.fit(
    data_aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=7
)

# Save the trained model
model.save('./recognizer/face_recognizer_resnet50.h5')
print("Training completed and model saved.")
'''
