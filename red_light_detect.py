import sounddevice as sd
import numpy as np
import librosa
import joblib
from datetime import datetime, timedelta
import threading

# Load the pre-trained model
model = joblib.load('audio_classifier.pkl')

# Audio configuration
CHUNK = 4096  # Number of samples per chunk
CONFIDENCE_THRESHOLD = 0.74  # Minimum confidence required for detection
DETECTION_WINDOW = 1.5  # Detection window in seconds for 3 detections
DELAY_AFTER_DETECTION = 0.2  # Delay in seconds before printing "red light"

# Detection state
detection_times = []

# Function to list available audio devices with 2 input channels
def list_audio_devices():
    print("Available audio devices with 2 input channels:")
    devices = sd.query_devices()
    compatible_devices = [
        (i, device['name'])
        for i, device in enumerate(devices)
        if device['max_input_channels'] >= 2  # At least 2 input channels
    ]
    for index, name in compatible_devices:
        print(f"[{index}] {name} (Input Channels: {sd.query_devices(index)['max_input_channels']})")
    return compatible_devices

# Function to get user-selected device index and supported sample rate
def get_device_index_and_rate():
    devices = list_audio_devices()
    default_index = 73

    # Check if default device is compatible
    if any(device[0] == default_index for device in devices):
        print(f"Using default device [73]: {sd.query_devices(default_index)['name']}")
        sample_rate = int(sd.query_devices(default_index)['default_samplerate'])
        return default_index, sample_rate

    print("Default device [73] not found or not compatible.")
    while True:
        try:
            device_index = int(input("Enter the index of the audio device to use: "))
            if any(device_index == d[0] for d in devices):
                sample_rate = int(sd.query_devices(device_index)['default_samplerate'])
                print(f"Selected device: {sd.query_devices(device_index)['name']} with sample rate: {sample_rate}")
                return device_index, sample_rate
            else:
                print("Invalid device index. Please select a listed compatible device.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to extract features from audio
def extract_features_from_chunk(audio_chunk, rate):
    try:
        # Ensure audio data is finite and normalized
        if not np.all(np.isfinite(audio_chunk)):
            raise ValueError("Audio buffer contains non-finite values")
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk) + 1e-6)  # Normalize
        # Compute MFCC features
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=rate, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# Function to check for 3 detections within 1 second
def check_detection_window():
    global detection_times
    now = datetime.now()
    # Remove detections older than 1 second
    detection_times = [t for t in detection_times if now - t < timedelta(seconds=DETECTION_WINDOW)]
    if len(detection_times) >= 3:
        threading.Timer(DELAY_AFTER_DETECTION, lambda: print("red light")).start()
        detection_times = []  # Reset after triggering

# Function to detect target audio in real-time
def detect_audio_stream(device_index, sample_rate):
    def callback(indata, frames, time, status):
        global detection_times
        if status:
            print(f"Status: {status}")
        audio_data = indata[:, 0]  # Use the first channel for mono input
        features = extract_features_from_chunk(audio_data, sample_rate)
        if features is not None:
            # Get prediction probabilities (confidence scores)
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[1]  # Confidence for class 1 (target audio)
            if confidence >= CONFIDENCE_THRESHOLD:
                detection_times.append(datetime.now())
                check_detection_window()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Detected with confidence: {confidence:.2f}")

    print("Listening for target audio...")
    with sd.InputStream(device=device_index, channels=2, samplerate=sample_rate, blocksize=CHUNK, callback=callback):
        input("Press Enter to stop...\n")

# Get the audio device and sample rate
selected_device_index, sample_rate = get_device_index_and_rate()

# Start audio stream detection
try:
    detect_audio_stream(selected_device_index, sample_rate)
except KeyboardInterrupt:
    print("Stopped listening.")
except Exception as e:
    print(f"Error: {e}")
