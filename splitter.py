import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def split_audio(file_path, output_folder, chunk_length_ms=3000):
    """
    Splits an audio file into chunks of specified length and saves them to a folder.
    
    :param file_path: Path to the audio file
    :param output_folder: Folder to save the chunks
    :param chunk_length_ms: Length of each chunk in milliseconds (default: 5000ms)
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Make chunks
    chunks = make_chunks(audio, chunk_length_ms)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each chunk to the output folder
    for i, chunk in enumerate(chunks):
        chunk_name = os.path.join(output_folder, f"chunk_{i+1}.wav")
        chunk.export(chunk_name, format="wav")
        print(f"Exported {chunk_name}")

# Example usage
file_path = "test_sample.wav"  # Replace with your audio file path
output_folder = "splits"  # Replace with your desired output folder
split_audio(file_path, output_folder)
