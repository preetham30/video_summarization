import gradio as gr
import requests
import os
import uuid
import tempfile
import time
from moviepy.editor import VideoFileClip
import whisper
from pydub import AudioSegment
import cv2
from tqdm import tqdm

QWEN_API_URL = "http://localhost:8001/v1/chat/completions"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

def call_qwen_vision(video_url: str) -> str:
    if not video_url.lower().startswith("http"):
        return "(Invalid video URL. Must be HTTP/S.)"

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": video_url}}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            return f"(Qwen HPC error {resp.status_code}: {resp.text})"
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"].get("content", "(No content found)")
        else:
            return "(No valid data from HPC.)"
    except Exception as e:
        return f"(HPC connection error: {e})"

def download_mp4(video_url: str) -> str:
    if not video_url.lower().startswith("http"):
        raise ValueError("Video URL must start with http or https.")
    tmp_path = os.path.join(tempfile.gettempdir(), f"vid_{uuid.uuid4()}.mp4")
    try:
        r = requests.get(video_url, stream=True, timeout=300)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return tmp_path
    except Exception as e:
        raise RuntimeError(f"Failed to download MP4 from {video_url}. Error: {e}")

def extract_audio_to_mp3(local_mp4_path: str) -> str:
    mp3_path = os.path.join(tempfile.gettempdir(), f"aud_{uuid.uuid4()}.mp3")
    try:
        clip = VideoFileClip(local_mp4_path)
        clip.audio.write_audiofile(mp3_path, fps=44100)
        clip.close()
        return mp3_path
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {e}")

def transcribe_mp3_whisper(mp3_path: str, model_size: str = "small", chunk_ms: int = 60000) -> str:
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        return f"(Error loading Whisper model '{model_size}': {e})"

    try:
        audio = AudioSegment.from_file(mp3_path)
    except Exception as e:
        return f"(Error reading MP3 file: {e})"

    total_len = len(audio)
    transcripts = []
    start_time = time.time()

    for start in range(0, total_len, chunk_ms):
        end = min(start + chunk_ms, total_len)
        chunk_audio = audio[start:end]
        tmp_chunk = os.path.join(tempfile.gettempdir(), f"chunk_{uuid.uuid4()}.mp3")
        chunk_audio.export(tmp_chunk, format="mp3")

        try:
            result = model.transcribe(tmp_chunk)
            text = result["text"]
        except Exception as e:
            text = f"(Error transcribing chunk: {e})"

        os.remove(tmp_chunk)
        sec_start = start // 1000
        sec_end = end // 1000
        transcripts.append(f"[{sec_start}-{sec_end}s] {text}")

    elapsed = time.time() - start_time
    combined = "\n".join(transcripts)
    return f"Whisper Transcript (model={model_size}, time={elapsed:.1f}s):\n{combined}"

def extract_frames_with_timestamps(local_mp4_path: str, desired_fps: int = 1) -> str:
    cap = cv2.VideoCapture(local_mp4_path)
    if not cap.isOpened():
        return "(Error: unable to open downloaded mp4 for frame extraction.)"

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    frames_info = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = int(round(original_fps / desired_fps)) if desired_fps > 0 else 1
    cur_frame = 0

    progress = tqdm(total=frame_count, desc="Frames", unit="frame", mininterval=1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur_frame % skip == 0:
            ts = cur_frame / original_fps
            frames_info.append(f"Frame={cur_frame}, time={ts:.2f}s")
        cur_frame += 1
        progress.update(1)
    progress.close()
    cap.release()
    return "\n".join(frames_info)

def merge_summaries(vision_text: str, audio_text: str, frames_info: str) -> str:
    merged = (
        f"=== Qwen2.5-VL Vision Output ===\n{vision_text}\n\n"
        f"=== Whisper Audio Transcript ===\n{audio_text}\n\n"
        f"=== Frame Timestamps ===\n{frames_info}\n"
    )
    return merged

def process_video_http(url: str, model_size: str = "small", chunk_sec: int = 60, desired_fps: int = 1):
    qwen_text = call_qwen_vision(url.strip())

    try:
        local_mp4 = download_mp4(url.strip())
    except Exception as e:
        local_mp4 = None
        qwen_text += f"\n(Error downloading MP4 for local processing: {e})"

    if local_mp4 and os.path.exists(local_mp4):
        try:
            mp3_file = extract_audio_to_mp3(local_mp4)
            audio_text = transcribe_mp3_whisper(mp3_file, model_size=model_size, chunk_ms=chunk_sec * 1000)
            frames_info = extract_frames_with_timestamps(local_mp4, desired_fps=desired_fps)
            os.remove(mp3_file)
        except Exception as e
::contentReference[oaicite:0]{index=0}
 

