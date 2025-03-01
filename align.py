import os
import json

def load_image_summaries(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def load_audio_summaries(json_file):
    with open(json_file, "r") as f:
        transcriptions = json.load(f)
    audio_summaries = {}
    for i, entry in enumerate(transcriptions, start=1):
        audio_summaries[i] = entry["text"]
    return audio_summaries

def assign_image_timestamps(image_summaries, frame_rate=30):
    image_timestamps = {}
    for frame_name, summary in image_summaries.items():
        frame_number = int(frame_name.split("_")[1].split(".")[0]) * 120
        timestamp = frame_number / frame_rate
        image_timestamps[timestamp] = summary
    return image_timestamps

def align_summaries(audio_summaries, image_timestamps, chunk_duration=60):
    aligned_summaries = {}
    for chunk_number, audio_summary in audio_summaries.items():
        if chunk_number < 1:
            continue
        start_time = (chunk_number - 1) * chunk_duration
        end_time = chunk_number * chunk_duration

        image_summaries_in_chunk = []
        for timestamp, image_summary in image_timestamps.items():
            if start_time <= timestamp < end_time:
                image_summaries_in_chunk.append(image_summary)

        aligned_summaries[f"{start_time}-{end_time} sec"] = {
            "audio_summary": audio_summary,
            "image_summaries": image_summaries_in_chunk,
        }
    return aligned_summaries

def generate_final_summary(aligned_summaries):
    final_summary = {}
    for time_segment, summaries in aligned_summaries.items():
        audio_summary = summaries["audio_summary"]
        image_summaries = summaries["image_summaries"]
        final_summary[time_segment] = {
            "Audio Summary": audio_summary,
            "Image Summaries": image_summaries
        }
    return final_summary

def save_final_summary(final_summary, output_file):
    with open(output_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"Final summary saved to {output_file}")

if __name__ == "__main__":
    image_json_file = "results.json"
    audio_json_file = "transcriptions.json"
    output_file = "final_summary.json"
    
    image_summaries = load_image_summaries(image_json_file)
    audio_summaries = load_audio_summaries(audio_json_file)
    image_timestamps = assign_image_timestamps(image_summaries)
    aligned_summaries = align_summaries(audio_summaries, image_timestamps)
    final_summary = generate_final_summary(aligned_summaries)
    save_final_summary(final_summary, output_file)

