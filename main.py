import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import librosa
import librosa.display
import os
import cv2

# Load MRI Image Data
def load_mri_images(image_folder, image_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, label)
        for file in os.listdir(class_folder):
            img = cv2.imread(os.path.join(class_folder, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load Speech Data and Extract Features
def extract_audio_features(audio_folder):
    features = []
    labels = []
    for label in os.listdir(audio_folder):
        class_folder = os.path.join(audio_folder, label)
        for file in os.listdir(class_folder):
            y, sr = librosa.load(os.path.join(class_folder, file))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.append(np.mean(mfcc, axis=1))
            labels.append(label)
    return np.array(features), np.array(labels)

# Load Clinical Data
def load_clinical_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    return X, y

# Prepare MRI Data (CNN Model)
mri_images, mri_labels = load_mri_images('mri_dataset')
mri_images = mri_images / 255.0  # Normalize
mri_images = mri_images.reshape(-1, 128, 128, 1)

# Encode Labels
label_encoder = LabelEncoder()
mri_labels = label_encoder.fit_transform(mri_labels)

X_train, X_test, y_train, y_test = train_test_split(mri_images, mri_labels, test_size=0.2, random_state=42)

# CNN Model for MRI
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(set(mri_labels)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Prepare Speech Data (NLP Model)
speech_features, speech_labels = extract_audio_features('speech_dataset')
speech_labels = label_encoder.fit_transform(speech_labels)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(speech_features, speech_labels, test_size=0.2, random_state=42)

# Train Random Forest for Speech Data
rf_speech = RandomForestClassifier(n_estimators=100)
rf_speech.fit(X_train_s, y_train_s)
preds_speech = rf_speech.predict(X_test_s)
print("Speech Model Accuracy:", accuracy_score(y_test_s, preds_speech))

# Prepare Clinical Data (ML Model)
X_clinical, y_clinical = load_clinical_data('clinical_data.csv')
X_clinical = StandardScaler().fit_transform(X_clinical)
y_clinical = label_encoder.fit_transform(y_clinical)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clinical, y_clinical, test_size=0.2, random_state=42)

# Train Random Forest for Clinical Data
rf_clinical = RandomForestClassifier(n_estimators=100)
rf_clinical.fit(X_train_c, y_train_c)
preds_clinical = rf_clinical.predict(X_test_c)
print("Clinical Model Accuracy:", accuracy_score(y_test_c, preds_clinical))
