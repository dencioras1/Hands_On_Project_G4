import os
import pickle
import seaborn as sns
import keras
from keras import ops
import librosa
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import Sequential
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import librosa.display
import random


class GenreClassifier:
    def __init__(self):
        self.label_encoder = None
        self.base_dir = "Audio"
        self.genres = sorted(os.listdir(self.base_dir))
        self.labels = ["BoomBap", "BossaNova", "BrazilianFunk", "Dancehall", "DnB", "Dubstep", "House", "JerseyClub",
                       "Reggaeton", "Trap"]
        self.model_path = "saved_models/model1.keras"
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.validation_split = None

    keras.utils.set_random_seed(42)

    def save_model(self, model_local):
        model_local.save(self.model_path)

    # Spectrogram extraction function
    def extract_mel_spectrogram(self, file_path, n_mels=128, fixed_length=330, input=input):
        y, sr = librosa.load(file_path, duration=8)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_clean = np.where(mel_db < -40, -80, mel_db)  # remove anything below -40dB
        mel_db = librosa.util.fix_length(mel_db_clean, size=fixed_length, axis=1)
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

    def show_spectogram(self, spectrogram):
        sr = 22050
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        plt.title(f"Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    def train_split(self, X, y):
        X = X[..., np.newaxis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def model_function(self):
        model = keras.models.Sequential([
            keras.Input(shape=(128, 330, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.05),

            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def train_model(self, model, X_test, y_test, X_train, y_train):
        # Set your hyperparameters
        # epochs = 8  # used to be 8  # Try different numbers 12
        # batch_size = 16  # Try different sizes 128
        optimizer = "adam"  # Try different optimizers "adam"
        # validation_split = 0.2  # Try different splits 0.2
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        model_history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                  validation_split=self.validation_split, callbacks=[early_stopping],
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

    def load_model(self):
        if self.model is None:
            print("Loading model")
            self.model = keras.models.load_model(self.model_path)
        else:
            print("We already have a model dumbass")

    def predict_file(self, file_path):
        spectrogram = self.extract_mel_spectrogram(file_path)
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-7)
        spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(spectrogram)
        index = np.argmax(prediction, axis=1)[0]
        return self.labels[index], prediction

    def show_model_training(self, model_history_local, model, X_test_local, y_test_local):
        # Plot Training and Validation Loss
        plt.figure(figsize=(5, 5))
        plt.plot(model_history_local.history['loss'], label='Train Loss', color='orange')
        plt.plot(model_history_local.history['val_loss'], label='Validation Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        # Plot Training and Validation Accuracy
        plt.figure(figsize=(5, 5))
        plt.plot(model_history_local.history['accuracy'], label='Train Accuracy', color='pink')
        plt.plot(model_history_local.history['val_accuracy'], label='Validation Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test_local, y_test_local)

        # Print test accuracy
        print("Test Accuracy:", test_accuracy)

    def confusionmatrix(self, model, X_test, y_test):
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(y_test, predicted_classes)

        sns.heatmap(conf_matrix, annot=True, fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title('Confusion Matrix')
        plt.show()

# GenreClassifier = GenreClassifier()

def grid_search_CNN():
    epoch_list = (6, 8, 10)
    batch_list = (12, 16, 32)
    validation_list = (0.3, 0.2, 0.1)
    all_val = []
    all_tr = []
    for a in epoch_list:
        for b in batch_list:
            for c in validation_list:
                GenreClassifier.epochs = a
                GenreClassifier.batch_size = b
                GenreClassifier.validation_split = c
                print(f"epochs: {a}, batch size: {b}, validation split: {c}")
                this_model_history, this_model, this_X_test, this_y_test = GenreClassifier.run_classifier()
                final_train_acc = this_model_history.history['accuracy'][-1]
                final_val_acc = this_model_history.history['val_accuracy'][-1]
                all_tr.append(final_train_acc)
                print(f"Final Training Accuracy: {final_train_acc:.4f}")
                all_val.append(final_val_acc)
                print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    all_val = sorted(all_val, reverse=True)[:5]
    print(f"Best validation scores: {all_val}")
    all_tr = sorted(all_tr, reverse=True)[:5]
    print(f"Best testing scores: {all_tr}")

def main():
    GenreClassifier.epochs = 6
    GenreClassifier.batch_size = 32
    GenreClassifier.validation_split = 0.1

    keras.utils.set_random_seed(42)

    # epochs: 8, batch size: 32, validation split: 0.1
    # Test Accuracy: 0.9166666865348816
    # Final Training Accuracy: 1.0000
    # Final Validation Accuracy: 1.0000
    #
    # epochs: 10, batch size: 12, validation split: 0.3
    # Test Accuracy: 0.9583333134651184
    # Final Training Accuracy: 1.0000
    # Final Validation Accuracy: 0.8966
    # epochs: 10, batch size: 12, validation split: 0.2

    # epochs: 6, batch size: 32, validation split: 0.1

    spect = GenreClassifier.extract_mel_spectrogram("output.wav")
    spect2 = GenreClassifier.extract_mel_spectrogram("Audio/BrazilianFunk/BrazilianFunkQuantized.wav")
    spect3 = GenreClassifier.extract_mel_spectrogram("Audio/House/HouseOffset #3.wav")
    GenreClassifier.show_spectogram(spect)
    GenreClassifier.show_spectogram(spect2)
    GenreClassifier.show_spectogram(spect3)

    this_model_history, this_model, this_X_test, this_y_test = GenreClassifier.run_classifier()
    # GenreClassifier.save_model(this_model)

    GenreClassifier.show_model_training(model_history_local=this_model_history, model=this_model,
                                        X_test_local=this_X_test,
                                        y_test_local=this_y_test)
    GenreClassifier.confusionmatrix(model=this_model, X_test=this_X_test, y_test=this_y_test)


# main()
