import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process a folder of audio files
def process_folder(folder, label):
    features, labels = [], []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        features.append(extract_features(file_path))
        labels.append(label)
    # Remove any None features (e.g., due to file errors)
    valid_data = [(f, l) for f, l in zip(features, labels) if f is not None]
    return zip(*valid_data)

# Process positive and negative folders
positive_features, positive_labels = process_folder('train/positive/', 1)
negative_features, negative_labels = process_folder('train/negative/', 0)

# Combine data
X = np.array(positive_features + negative_features)
y = np.array(positive_labels + negative_labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'audio_classifier.pkl')

# Test the model on new audio
def detect_audio(file_path, model):
    features = extract_features(file_path)
    if features is not None:
        prediction = model.predict([features])
        return bool(prediction[0])
    else:
        print("Failed to extract features from the audio file.")
        return None

# Load the model and test on a new file
loaded_model = joblib.load('audio_classifier.pkl')
result = detect_audio('new_audio.wav', loaded_model)
print("Detected target audio:", result)
