import whisper
import os
import tempfile
import json
import time
from pydub import AudioSegment

def load_whisper_model(model_size="base"):
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    return model

def transcribe_audio(model, audio_path):
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)
    return result["text"]

def process_audio_in_chunks(model, audio_path, chunk_length_ms=60000, output_dir="transcriptions"):
    audio = AudioSegment.from_file(audio_path)
    total_length = len(audio)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Total Audio Length: {total_length / 1000:.2f} seconds")
    print(f"Processing in {chunk_length_ms / 1000:.2f}-second chunks...\n")

    transcriptions = []
    start_index = 0

    for i, start_time in enumerate(range(0, total_length, chunk_length_ms), start=1):
        end_time = min(start_time + chunk_length_ms, total_length)
        chunk = audio[start_time:end_time]

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            chunk.export(temp_file.name, format="mp3")
            transcription = transcribe_audio(model, temp_file.name)
            transcription_data = {
                "time_stamp": f"{start_time // 1000}-{end_time // 1000}s",
                "text": transcription
            }
            transcriptions.append(transcription_data)
            os.remove(temp_file.name)

    output_file = os.path.join(output_dir, "transcriptions.json")
    with open(output_file, "w") as f:
        json.dump(transcriptions, f, indent=4)

    print(f"Transcriptions saved to {output_file}")
    return transcriptions

if __name__ == "__main__":
    start_time = time.time()
    audio_path = "extracted_audio.mp3"
    model = load_whisper_model(model_size="tiny")
    trans_dir = "transcriptions"
    transcriptions = process_audio_in_chunks(model, audio_path, chunk_length_ms=60000, output_dir=trans_dir)

    print("\nTranscriptions:")
    for entry in transcriptions:
        print(json.dumps(entry, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"Total time: {hours}h {minutes}m {seconds:.2f}s")

