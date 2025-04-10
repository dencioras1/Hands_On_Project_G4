import os
import pickle

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


class GenreClassifier:
    def __init__(self):
        self.label_encoder = None
        self.base_dir = "/Users/kveen/Documents/GitHub/Hands_On_Project_G4/Audio"
        self.genres = sorted(os.listdir(self.base_dir))



    # Spectrogram extraction function
    def extract_mel_spectrogram(self, file_path, n_mels=128, fixed_length=330):
        y, sr = librosa.load(file_path, duration=8)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=fixed_length, axis=1)
        return mel_db


    def data_prep(self):
        # Make sure these are lists
        X = []
        y = []

        # Loop through genres and files
        for genre in tqdm(self.genres):
            genre_dir = os.path.join(self.base_dir, genre)

            # Skip .DS_Store or anything that's not a folder
            if not os.path.isdir(genre_dir):
                continue

            for filename in os.listdir(genre_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_dir, filename)
                    try:
                        spectrogram = self.extract_mel_spectrogram(file_path)
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
        return X, y


    def show_random_spectrogram(self, X, y, label_encoder=None, sr=22050, fixed_length=330):
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

    def train_split(self, X, y):
        X = X[..., np.newaxis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    def model_function(self):
        model = keras.models.Sequential([
            keras.Input(shape=(128, 330, 1)),
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Dropout(0.05),

            keras.layers.Conv2D(16, (3,3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2,2),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def train_model(self, model, X_test, y_test, X_train, y_train):
        # Set your hyperparameters
        epochs = 8  # Try different numbers 12
        batch_size = 16  # Try different sizes 128
        optimizer = "adam"  # Try different optimizers "adam"
        validation_split = 0.2 # Try different splits 0.2

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        model_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                                  verbose=1)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        print("Test Accuracy:", test_accuracy)
        return model_history

    def run_classifier(self):
        X, y = self.data_prep()
        X_train, X_test, y_train, y_test = self.train_split(X, y)
        model = self.model_function()
        model_history = self.train_model(model, X_test, y_test, X_train, y_train)
        # plot(model_history, model, X_test, y_test)
        # model.save("saved_models/genre_model.h5")
        # with open("saved_models/label_encoder.pkl", "wb") as f:
        #     pickle.dump(self.label_encoder, f)


        return model_history, model, X_test, y_test

    # def load_model_and_encoder(self, model_path="saved_models/genre_model.h5",
    #                            encoder_path="saved_models/label_encoder.pkl"):
    #     self.model = keras.models.load_model(model_path)
    #     with open(encoder_path, "rb") as f:
    #         self.label_encoder = pickle.load(f)
    #
    # def predict_file(self, file_path):
    #     spectrogram = self.extract_mel_spectrogram(file_path)
    #     spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-7)
    #     spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    #     prediction = self.model.predict(spectrogram)
    #     index = np.argmax(prediction, axis=1)[0]
    #     return self.label_encoder.inverse_transform([index])[0]


def plot(model_history, model, X_test, y_test):
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


# def gridsearch():
#     conv1 = (16, 32, 64)
#     conv2 = (16, 32, 64)
#     dropoutrate = (0.01, 0.05, 0.1)
#     epoch_size = (8, 12, 16)
#     batch_size = (16, 32, 64)
#     all_histories = []
#     all_values = []
#
#     for a in conv1:
#         for b in conv2:
#             for c in dropoutrate:
#                 for d in epoch_size:
#                     for e in batch_size:
#                         local_model = model(a, b, c)
#                         train_history = train_model(local_model, d, e)
#                         local_values = [a, b, c, d, e]
#                         all_histories.append(train_history)
#                         all_values.append(local_values)
#                         print(f"{a}, {b}, {c}, {d}, {e}")
#     return all_histories, all_values





# all_histories, all_values = gridsearch()

# Best values:
    #   16, 16, 0.05, 12, 32
    # 16, 16, 0.05, 12, 64 ; due to smaller batch size would be better for input of new data
    # 32, 16, 0.05, 8, 16
    # 32, 64, 0.1, 16, 64
    # ? 32, 32, 0.1, 12, 64
    # ? 64, 16, 0.1, 12, 64


# def main():
#     X, y = data_prep()
#     X_train, X_test, y_train, y_test = train_split(X, y)
#     model = model_function()
#     model_history = train_model(model, X_test, y_test, X_train, y_train)
#     show_random_spectrogram(X, y, label_encoder=None)
#     plot(model_history, model, X_test, y_test)

