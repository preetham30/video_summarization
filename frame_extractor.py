import cv2
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: No frames found. Invalid video file.")
        return

    print(f"Total Frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames.")

    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")

def extract_video_clips(video_path, output_folder, clip_duration=10, frame_rate=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    if total_frames == 0:
        print("Error: No frames found. Invalid video file.")
        return

    print(f"Total Frames: {total_frames}")
    print(f"Original Frame Rate: {original_frame_rate} fps")
    print(f"Extracting clips of {clip_duration} seconds each.")
    frames_per_clip = clip_duration * frame_rate

    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    frame_count = 0
    clip_count = 0

    while True:
        clip_filename = os.path.join(output_folder, f"clip_{clip_count:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_filename, fourcc, frame_rate, (frame_width, frame_height))

        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
            progress_bar.update(1)

        out.release()
        clip_count += 1
        if not ret:
            break

    progress_bar.close()
    cap.release()
    print(f"Extracted {clip_count} clips to {output_folder}")

def extract_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    print(f"Audio extracted and saved to {output_audio_path}")

if __name__ == "__main__":
    video_path = "sample_video_3.mp4"
    output_folder = "extracted_frames"
    output_audio_path = "extracted_audio.mp3"
    extract_frames(video_path, output_folder, frame_interval=120)
    extract_audio(video_path, output_audio_path)

