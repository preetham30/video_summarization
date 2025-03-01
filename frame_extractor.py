import cv2
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts 1 frame every `frame_interval` frames from a video.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save extracted frames.
        frame_interval (int): Extract 1 in `frame_interval` frames.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:  # Avoid processing empty/corrupt videos
        print("Error: No frames found. Invalid or corrupt video file.")
        return
    
    print(f"Total Frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames.")

    # Progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Save every `frame_interval` frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
        progress_bar.update(1)  # Update tqdm progress

    progress_bar.close()
    cap.release()

    print(f"Extracted {saved_frame_count} frames to {output_folder}")

def extract_video_clips(video_path, output_folder, clip_duration=10, frame_rate=30):
    """
    Extracts video clips of `clip_duration` seconds from a video.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save extracted video clips.
        clip_duration (int): Duration of each clip in seconds (default: 10 seconds).
        frame_rate (int): Frame rate of the video (default: 30 fps).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    if total_frames == 0:  # Avoid processing empty/corrupt videos
        print("Error: No frames found. Invalid or corrupt video file.")
        return
    
    print(f"Total Frames: {total_frames}")
    print(f"Original Frame Rate: {original_frame_rate} fps")
    print(f"Extracting clips of {clip_duration} seconds each.")

    # Calculate the number of frames per clip
    frames_per_clip = clip_duration * frame_rate

    # Progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    frame_count = 0
    clip_count = 0

    while True:
        # Create a new video writer for each clip
        clip_filename = os.path.join(output_folder, f"clip_{clip_count:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(clip_filename, fourcc, frame_rate, (frame_width, frame_height))

        # Write frames to the current clip
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            out.write(frame)
            frame_count += 1
            progress_bar.update(1)  # Update tqdm progress

        out.release()  # Close the current clip
        clip_count += 1

        if not ret:
            break  # End of video

    progress_bar.close()
    cap.release()

    print(f"Extracted {clip_count} clips to {output_folder}")

# Function to extract audio from a video
def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from a video and saves it as an MP3 file.
    
    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    print(f"Audio extracted and saved to {output_audio_path}")

# Example usage
if __name__ == "__main__":
    video_path = "sample_video_3.mp4"  # Input video path
    output_folder = "extracted_frames"  # Folder for extracted frames
    output_audio_path = "extracted_audio.mp3"  # Path for extracted audio
    
    # Extract 1 in every 50 frames
    extract_frames(video_path, output_folder, frame_interval=120)

    #extract_video_clips(video_path, output_folder, clip_duration=40, frame_rate=10)
    
    # Extract audio
    extract_audio(video_path, output_audio_path)
