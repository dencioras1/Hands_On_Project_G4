import os

import keras
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import random

base_dir = "/Users/kveen/Documents/GitHub/Hands_On_Project_G4/Audio"
genres = sorted(os.listdir(base_dir))

# Make sure these are lists
X = []
y = []

# Spectrogram extraction function
def extract_mel_spectrogram(file_path, n_mels=128, fixed_length=330):
    y, sr = librosa.load(file_path, duration=8)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=fixed_length, axis=1)
    return mel_db

# Loop through genres and files
for genre in tqdm(genres):
    genre_dir = os.path.join(base_dir, genre)

    # Skip .DS_Store or anything that's not a folder
    if not os.path.isdir(genre_dir):
        continue

    for filename in os.listdir(genre_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(genre_dir, filename)
            try:
                spectrogram = extract_mel_spectrogram(file_path)
                X.append(spectrogram)  # Make sure X is a list here
                y.append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert only after all .append() calls are done
X = np.array(X)
y = np.array(y)


# Create and fit the encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # y was a list/array of genres

# Show mapping
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

y = y_encoded
X = (X - np.mean(X)) / (np.std(X) + 1e-7)


print("Shape of X:", X.shape)
print("Labels:", np.unique(y))


def show_random_spectrogram(X, y, label_encoder=None, sr=22050, fixed_length=330):
    idx = random.randint(0, len(X) - 1)
    spectrogram = X[idx].squeeze()
    label = y[idx]

    # Ensure only first 660 frames (8 sec) are shown
    spectrogram = spectrogram[:, :fixed_length]

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    genre = label if label_encoder is None else label_encoder.inverse_transform([label])[0]
    plt.title(f"Mel Spectrogram - Genre: {genre}")
    plt.tight_layout()
    plt.show()


show_random_spectrogram(X, y, label_encoder=None)
X = X[..., np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def model():
    model = keras.models.Sequential([
        keras.Input(shape=(128, 330, 1)),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(model):


    # Set your hyperparameters
    epochs = 12  # Try different numbers 12
    batch_size = 64  # Try different sizes 128
    optimizer = "adam"  # Try different optimizers "adam"
    validation_split = 0.20 # Try different splits 0.2

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("Test Accuracy:", test_accuracy)
    return model_history

def plot(model_history):
    # Plot Training and Validation Loss
    plt.figure(figsize=(5, 5))
    plt.plot(model_history.history['loss'], label='Train Loss', color='orange')
    plt.plot(model_history.history['val_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(5, 5))
    plt.plot(model_history.history['accuracy'], label='Train Accuracy', color='pink')
    plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Print test accuracy
    print("Test Accuracy:", test_accuracy)

model = model()
model_history = train_model(model)
plot(model_history)