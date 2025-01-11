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
result = detect_audio('red7.wav', loaded_model)
print("Detected target audio:", result)
