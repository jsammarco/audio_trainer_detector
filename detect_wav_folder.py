import librosa
import numpy as np
import os
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

# Function to test the model on a new audio file
def detect_audio(file_path, model):
    features = extract_features(file_path)
    if features is not None:
        prediction = model.predict([features])
        return bool(prediction[0])
    else:
        print(f"Failed to extract features from the audio file: {file_path}")
        return None

# Function to process all .wav files in a folder
def process_folder(folder_path, model):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            result = detect_audio(file_path, model)
            print(f"File: {file_name}, Detected: {result}")

# Load the trained model
loaded_model = joblib.load('audio_classifier.pkl')

# Folder containing .wav files to check
folder_path = 'samples'  # Replace with the actual folder path

# Process the folder and print results
process_folder(folder_path, loaded_model)
