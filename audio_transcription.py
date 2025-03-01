import whisper
import os
import tempfile
import json
import time
from pydub import AudioSegment

def load_whisper_model(model_size="base"):
    """
    Load the Whisper model.
    
    Args:
        model_size (str): Size of the Whisper model. Options: "tiny", "base", "small", "medium", "large".
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    return model

def transcribe_audio(model, audio_path):
    """
    Transcribe an audio file using Whisper.
    
    Args:
        model: Loaded Whisper model.
        audio_path (str): Path to the audio file.
    
    Returns:
        str: Transcribed text.
    """
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)
    return result["text"]

def process_audio_in_chunks(model, audio_path, chunk_length_ms=60000, output_dir="transcriptions"):
    """
    Process an audio file in smaller chunks efficiently to avoid memory overload.
    
    Args:
        model: Loaded Whisper model.
        audio_path (str): Path to the audio file.
        chunk_length_ms (int): Length of each chunk in milliseconds (default: 1 minute).
        output_dir (str): Directory to save transcribed chunks.
    
    Returns:
        list: List of dictionaries containing chunked transcriptions in JSON format.
    """
    # Load the audio
    audio = AudioSegment.from_file(audio_path)
    total_length = len(audio)


    print(f"Total Audio Length: {total_length / 1000:.2f} seconds")
    print(f"Processing in {chunk_length_ms / 1000:.2f}-second chunks...\n")

    transcriptions = []

    for i, start_time in enumerate(range(0, total_length, chunk_length_ms)):
        end_time = min(start_time + chunk_length_ms, total_length)
        chunk = audio[start_time:end_time]

        # Save chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            chunk.export(temp_file.name, format="mp3")
            transcription = transcribe_audio(model, temp_file.name)
            
            # Store transcription in JSON format
            transcription_data = {
                "time_stamp": f"{start_time // 1000}-{end_time // 1000}s",
                "text": transcription
            }
            transcriptions.append(transcription_data)

            # Remove the temporary chunk file
            os.remove(temp_file.name)

    # Save transcriptions to a JSON file
    output_file = "transcriptions.json"
    with open(output_file, "w") as f:
        json.dump(transcriptions, f, indent=4)

    print(f"Transcriptions saved to {output_file}")
    return transcriptions

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    audio_path = "extracted_audio.mp3"  # Replace with your audio file path
    model = load_whisper_model(model_size="tiny")  # Options: "tiny", "base", "small", "medium", "large"
    
    # Process and transcribe audio efficiently in chunks and store transcriptions in JSON format
    transcriptions = process_audio_in_chunks(model, audio_path, chunk_length_ms=15000, output_dir="")
    
    print("\nTranscriptions:")
    for entry in transcriptions:
        print(json.dumps(entry, indent=4))

    end_time = time.time()
    total_time = end_time - start_time

    # Convert total time into hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # Print human-readable time
    print(f"Total time taken: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")