import os
import json

# Load image summaries from JSON file
def load_image_summaries(json_file):
    """
    Load image summaries from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing image summaries.
    
    Returns:
        dict: Dictionary of image summaries with frame names as keys.
    """
    with open(json_file, "r") as f:
        return json.load(f)

# Load audio summaries from transcriptions.json
def load_audio_summaries(json_file):
    """
    Load audio summaries from a JSON file containing transcriptions.
    
    Args:
        json_file (str): Path to the JSON file containing audio transcriptions.
    
    Returns:
        dict: Dictionary of audio summaries with chunk IDs as keys.
    """
    with open(json_file, "r") as f:
        transcriptions = json.load(f)
    
    audio_summaries = {}
    for i, entry in enumerate(transcriptions, start=1):
        audio_summaries[i] = entry["text"]  # Use chunk ID (i) as the key
    
    return audio_summaries

# Assign timestamps to image summaries
def assign_image_timestamps(image_summaries, frame_rate=30):
    """
    Assign timestamps to image summaries based on frame numbers and frame rate.
    
    Args:
        image_summaries (dict): Dictionary of image summaries with frame names as keys.
        frame_rate (int): Frame rate of the video (default: 30 fps).
    
    Returns:
        dict: Dictionary of image summaries with timestamps as keys.
    """
    image_timestamps = {}
    for frame_name, summary in image_summaries.items():
        # Extract the frame number from the filename and multiply by 100
        frame_number = int(frame_name.split("_")[1].split(".")[0]) * 120
        timestamp = frame_number / frame_rate  # Time in seconds
        image_timestamps[timestamp] = summary
    return image_timestamps

# Align audio and image summaries by time
def align_summaries(audio_summaries, image_timestamps, chunk_duration=60):
    """
    Align audio and image summaries by time.
    
    Args:
        audio_summaries (dict): Dictionary of audio summaries with chunk IDs as keys.
        image_timestamps (dict): Dictionary of image summaries with timestamps as keys.
        chunk_duration (int): Duration of each audio chunk in seconds (default: 60).
    
    Returns:
        dict: Dictionary of aligned summaries with time segments as keys.
    """
    aligned_summaries = {}
    
    for chunk_number, audio_summary in audio_summaries.items():
        # Ensure chunk_number starts from 1
        if chunk_number < 1:
            continue  # Skip invalid chunk numbers
        
        start_time = (chunk_number - 1) * chunk_duration
        end_time = chunk_number * chunk_duration
        
        # Find image summaries within this time range
        image_summaries_in_chunk = []
        for timestamp, image_summary in image_timestamps.items():
            if start_time <= timestamp < end_time:
                image_summaries_in_chunk.append(image_summary)
        
        # Combine audio and image summaries
        aligned_summaries[f"{start_time}-{end_time} sec"] = {
            "audio_summary": audio_summary,
            "image_summaries": image_summaries_in_chunk,
        }
    
    return aligned_summaries

# Generate a final summary for each time segment
def generate_final_summary(aligned_summaries):
    """
    Generate a final summary for each time segment by combining audio and image summaries.
    
    Args:
        aligned_summaries (dict): Dictionary of aligned summaries with time segments as keys.
    
    Returns:
        dict: Dictionary of final summaries with time segments as keys.
    """
    final_summary = {}
    
    for time_segment, summaries in aligned_summaries.items():
        audio_summary = summaries["audio_summary"]
        image_summaries = summaries["image_summaries"]
        
        # Store audio and image summaries as separate key-value pairs
        final_summary[time_segment] = {
            "Audio Summary": audio_summary,
            "Image Summaries": image_summaries
        }
    
    return final_summary

# Save final summary to a JSON file
def save_final_summary(final_summary, output_file):
    """
    Save the final summary to a JSON file.
    
    Args:
        final_summary (dict): Dictionary of final summaries with time segments as keys.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"Final summary saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Paths to input files
    image_json_file = "results.json"  # Replace with your JSON file path
    audio_json_file = "transcriptions.json"  # Replace with your JSON file path for audio transcriptions
    output_file = "final_summary.json"  # Output file for the final summary
    
    # Load summaries
    image_summaries = load_image_summaries(image_json_file)
    audio_summaries = load_audio_summaries(audio_json_file)
    
    # Assign timestamps to image summaries
    image_timestamps = assign_image_timestamps(image_summaries)
    
    # Align audio and image summaries by time
    aligned_summaries = align_summaries(audio_summaries, image_timestamps)
    
    # Generate final summary
    final_summary = generate_final_summary(aligned_summaries)
    
    # Save final summary to a JSON file
    save_final_summary(final_summary, output_file)