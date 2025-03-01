import requests
import os
import uuid
import tempfile
import time
import cv2
from tqdm import tqdm
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import re

# HPC endpoint – adjust as needed.
QWEN_API_URL = "http://localhost:8001/v1/chat/completions"

def call_qwen_vision(prompt: str, video_url: str) -> str:
    """Call Qwen2.5-VL HPC with the given prompt and video URL."""
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            return f"[Error] HPC error {resp.status_code}: {resp.text}"
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"].get("content", "[Error] No HPC content found")
        else:
            return "[Error] No valid HPC data returned."
    except Exception as e:
        return f"[Error] HPC connection error: {e}"

def download_mp4(video_url: str) -> str:
    """Download the MP4 from the provided URL to a temporary file."""
    if not video_url.lower().startswith("http"):
        raise ValueError("Video URL must start with http or https.")
    tmp_path = os.path.join(tempfile.gettempdir(), f"vid_{uuid.uuid4()}.mp4")
    r = requests.get(video_url, stream=True, timeout=300)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path

def extract_audio_to_mp3(local_mp4: str) -> str:
    """Extract audio from the local MP4 and save it as an MP3."""
    mp3_path = os.path.join(tempfile.gettempdir(), f"aud_{uuid.uuid4()}.mp3")
    clip = VideoFileClip(local_mp4)
    clip.audio.write_audiofile(mp3_path, fps=44100, logger=None)
    clip.close()
    return mp3_path

def transcribe_mp3_whisper(mp3_path: str, model_size: str = "small", chunk_ms: int = 60000) -> str:
    """Transcribe the MP3 using Whisper (in chunks) and return the transcript with timestamps."""
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        return f"[Error] Loading Whisper model '{model_size}': {e}"
    
    try:
        audio = AudioSegment.from_file(mp3_path)
    except Exception as e:
        return f"[Error] Reading MP3 file: {e}"
    
    total_ms = len(audio)
    transcripts = []
    start_time = time.time()
    for start in range(0, total_ms, chunk_ms):
        end = min(start + chunk_ms, total_ms)
        chunk = audio[start:end]
        tmp_chunk = os.path.join(tempfile.gettempdir(), f"chunk_{uuid.uuid4()}.mp3")
        chunk.export(tmp_chunk, format="mp3")
        try:
            result = model.transcribe(tmp_chunk)
            text = result.get("text", "")
        except Exception as e:
            text = f"[Error] Transcription: {e}"
        finally:
            if os.path.exists(tmp_chunk):
                os.remove(tmp_chunk)
        transcripts.append(f"[{start//1000}-{end//1000}s] {text}")
    elapsed = time.time() - start_time
    return f"Whisper Transcript (model={model_size}, time={elapsed:.1f}s):\n" + "\n".join(transcripts)

def extract_frames_with_timestamps(local_mp4: str, desired_fps: int = 1) -> str:
    """
    Extract frame timestamps from the MP4.
    For example, if the video is 30fps and desired_fps is 1, you'll get one timestamp per second.
    """
    cap = cv2.VideoCapture(local_mp4)
    if not cap.isOpened():
        return "[Error] Could not open MP4 for frame extraction."
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = int(round(orig_fps / desired_fps)) if desired_fps > 0 else 999999
    lines = []
    current_frame = 0
    progress = tqdm(total=total_frames, desc="Extracting Frames", unit="frame", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % skip == 0:
            ts = current_frame / orig_fps
            lines.append(f"Frame={current_frame}, time={ts:.2f}s")
        current_frame += 1
        progress.update(1)
    progress.close()
    cap.release()
    return "\n".join(lines)

def remove_timestamps(text: str) -> str:
    """Remove [start-end] markers from transcript lines."""
    return re.sub(r'\[[^\]]+\]', '', text)

def merge_summaries(vision_text: str, whisper_text: str, frames_info: str) -> str:
    """Merge vision output with the Whisper transcript (without timestamps) for final merged output."""
    clean_whisper = remove_timestamps(whisper_text).strip()
    merged = vision_text.strip() + "\n\n" + clean_whisper
    return merged

def process_video_once(prompt: str, url_or_path: str, model_size: str, chunk_sec: int, fps: int):
    """
    Run the entire pipeline once.
    If url_or_path starts with "http", it's treated as a URL; otherwise, as a local file.
    Returns (vision_text, whisper_text, merged_output).
    """
    vision_text = call_qwen_vision(prompt, url_or_path)
    if url_or_path.lower().startswith("http"):
        local_mp4 = download_mp4(url_or_path)
    else:
        local_mp4 = url_or_path
    mp3_file = extract_audio_to_mp3(local_mp4)
    whisper_text = transcribe_mp3_whisper(mp3_file, model_size=model_size, chunk_ms=chunk_sec*1000)
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
    if fps > 0:
        frames_info = extract_frames_with_timestamps(local_mp4, fps)
    else:
        frames_info = ""
    if url_or_path.lower().startswith("http") and os.path.exists(local_mp4):
        os.remove(local_mp4)
    merged_output = merge_summaries(vision_text, whisper_text, frames_info)
    return vision_text, whisper_text, merged_output

# Wrappers for different interfaces:
def process_chat(prompt: str, url: str, modality: str, model_size: str, chunk_sec: int, fps: int):
    if modality == "Text Only":
        vision_text = prompt
        whisper_text = ""
        merged = prompt
    elif modality == "Audio Only":
        vision_text = "[N/A]"
        vision_text, whisper_text, merged = process_video_once(prompt, url, model_size, chunk_sec, fps)
    else:
        vision_text, whisper_text, merged = process_video_once(prompt, url, model_size, chunk_sec, fps)
    return vision_text, whisper_text, merged

def process_screen_capture(file, model_size: str, chunk_sec: int, fps: int):
    if file is None:
        return "Error: No file uploaded.", "", ""
    return process_video_once("Summarize video", file.name, model_size, chunk_sec, fps)

def process_webcam(video_file, model_size: str, chunk_sec: int, fps: int):
    if video_file is None:
        return "Error: No webcam video recorded.", "", ""
    return process_video_once("Summarize video", video_file.name, model_size, chunk_sec, fps)

# For command-line testing:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Captioning via Qwen2.5-VL HPC and Whisper")
    parser.add_argument("--prompt", required=True, help="Prompt text (e.g., 'Summarize video')")
    parser.add_argument("--url", required=True, help="Remote MP4 URL or local file path")
    parser.add_argument("--whisper_model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--chunk_sec", type=int, default=60, help="Audio chunk length in seconds")
    parser.add_argument("--fps", type=int, default=1, help="Frame extraction rate (FPS; 0 to skip)")
    args = parser.parse_args()
    
    vision, whisper_transcript, merged = process_video_once(
        prompt=args.prompt,
        url_or_path=args.url,
        model_size=args.whisper_model,
        chunk_sec=args.chunk_sec,
        fps=args.fps
    )
    print("\n=== Vision Output ===\n")
    print(vision)
    print("\n=== Whisper Transcript ===\n")
    print(whisper_transcript)
    print("\n=== Merged Output ===\n")
    print(merged)

