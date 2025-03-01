import requests
import os
import uuid
import tempfile
import time
import cv2
import numpy as np
import pyaudio
import wave
import threading
import queue
import mss
from moviepy.editor import VideoFileClip

# ------------------------------------------------------------------
# Qwen Endpoints (do not change these URLs)
# ------------------------------------------------------------------
QWEN_VISION_API_URL = "http://localhost:8001/v1/chat/completions"
QWEN_AUDIO_API_URL = "http://localhost:8011/v1/audio/transcriptions"

# =========== Vision Inference ===========
def call_qwen_vision(prompt: str, file_path: str) -> str:
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": file_path}}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(QWEN_VISION_API_URL, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            return f"[Vision Error {resp.status_code}] {resp.text}"
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"].get("content", "[No vision output]")
        return "[No vision output]"
    except Exception as e:
        return f"[Vision Exception] {e}"

# =========== Audio Inference ===========
def transcribe_mp3_qwen(mp3_path: str, prompt: str = "Transcribe the audio.") -> str:
    try:
        with open(mp3_path, "rb") as f:
            files = {"file": f}
            data = {"model": "Qwen/Qwen2-Audio-7B-Instruct", "prompt": prompt}
            resp = requests.post(QWEN_AUDIO_API_URL, data=data, files=files, timeout=120)
        if resp.status_code != 200:
            return f"[Audio Error {resp.status_code}] {resp.text}"
        result = resp.json()
        transcription = result.get("transcription", "")
        if not transcription and "choices" in result:
            transcription = result["choices"][0]["message"].get("content", "")
        return transcription
    except Exception as e:
        return f"[Audio Exception] {e}"

# ------------------------------------------------------------------
# RealTimeProcessor: Real-Time Screen & Audio Capture with Merged Timestamps
# ------------------------------------------------------------------
class RealTimeProcessor:
    def __init__(self, prompt="Describe the screen and audio in real time",
                 capture_interval=5.0, audio_chunk_duration=5.0, frame_skip=10):
        self.prompt = prompt
        self.capture_interval = capture_interval
        self.audio_chunk_duration = audio_chunk_duration
        self.frame_skip = frame_skip

        self.running = False
        self._start_time = None

        self.latest_frame = None
        self._frame_lock = threading.Lock()

        self.vision_events = []   # List of {"timestamp": float, "content": str}
        self._vision_lock = threading.Lock()

        self.audio_events = []    # List of {"timestamp": float, "content": str}
        self._audio_lock = threading.Lock()

        self.timeline_merged = []  # Combined events
        self._timeline_lock = threading.Lock()

        self.sct = mss.mss()
        self.audio_q = queue.Queue()

        # Audio configuration
        self.rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk = 1024

        self.screen_thread = None
        self.audio_thread = None
        self.proc_thread = None

    def start(self):
        self.running = True
        self._start_time = time.time()
        self.screen_thread = threading.Thread(target=self._capture_screen, daemon=True)
        self.audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.screen_thread.start()
        self.audio_thread.start()
        self.proc_thread.start()

    def stop(self):
        self.running = False

    def _capture_screen(self):
        monitor = self.sct.monitors[1]
        skip_counter = 0
        while self.running:
            img = np.array(self.sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BG2RGB)
            skip_counter += 1
            if skip_counter >= self.frame_skip:
                skip_counter = 0
                with self._frame_lock:
                    self.latest_frame = frame
            time.sleep(0.04)

    def _capture_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        frames = []
        chunk_count = int((self.rate / self.chunk) * self.audio_chunk_duration)
        while self.running:
            data = stream.read(self.chunk)
            frames.append(data)
            if len(frames) >= chunk_count:
                wav_path = os.path.join(tempfile.gettempdir(), f"live_{uuid.uuid4()}.wav")
                wf = wave.open(wav_path, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                self.audio_q.put(wav_path)
                frames = []
        stream.stop_stream()
        stream.close()
        p.terminate()

    def _process_loop(self):
        last_vision_time = 0.0
        while self.running:
            now = time.time()
            elapsed = now - self._start_time

            # Vision processing
            if (now - last_vision_time) >= self.capture_interval:
                last_vision_time = now
                with self._frame_lock:
                    frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
                if frame_copy is not None:
                    png_path = os.path.join(tempfile.gettempdir(), f"frame_{uuid.uuid4()}.png")
                    cv2.imwrite(png_path, cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR))
                    vision_text = call_qwen_vision(self.prompt, png_path)
                    if os.path.exists(png_path):
                        os.remove(png_path)
                    with self._vision_lock:
                        self.vision_events.append({"timestamp": elapsed, "content": vision_text})
                    with self._timeline_lock:
                        self.timeline_merged.append({"timestamp": elapsed, "type": "VISION", "content": vision_text})

            # Audio processing
            try:
                wav_file = self.audio_q.get_nowait()
            except queue.Empty:
                wav_file = None
            if wav_file:
                mp3_path = os.path.join(tempfile.gettempdir(), f"live_{uuid.uuid4()}.mp3")
                try:
                    clip = VideoFileClip(wav_file)
                    clip.audio.write_audiofile(mp3_path, fps=16000, logger=None)
                    clip.close()
                except Exception:
                    mp3_path = wav_file
                audio_text = transcribe_mp3_qwen(mp3_path, self.prompt)
                if os.path.exists(wav_file) and wav_file != mp3_path:
                    os.remove(wav_file)
                if os.path.exists(mp3_path) and mp3_path != wav_file:
                    os.remove(mp3_path)
                now2 = time.time()
                elapsed2 = now2 - self._start_time
                with self._audio_lock:
                    self.audio_events.append({"timestamp": elapsed2, "content": audio_text})
                with self._timeline_lock:
                    self.timeline_merged.append({"timestamp": elapsed2, "type": "AUDIO", "content": audio_text})
            time.sleep(0.2)

    def get_latest_overlay(self):
        with self._frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None:
            return None, "", ""
        with self._vision_lock:
            last_vis = self.vision_events[-1]["content"] if self.vision_events else ""
        with self._audio_lock:
            last_aud = self.audio_events[-1]["content"] if self.audio_events else ""
        overlay_frame = frame.copy()
        overlay_lines = []
        if last_vis.strip():
            overlay_lines.append(f"VISION: {last_vis.strip()}")
        if last_aud.strip():
            overlay_lines.append(f"AUDIO: {last_aud.strip()}")
        overlay_text = "\n".join(overlay_lines)
        y0, dy = 30, 25
        for i, line in enumerate(overlay_text.split("\n")):
            y = y0 + i * dy
            cv2.putText(overlay_frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay_frame, last_aud, last_vis

    def get_merged_data(self):
        with self._timeline_lock:
            merged_sorted = sorted(self.timeline_merged, key=lambda ev: ev["timestamp"])
            lines = []
            for ev in merged_sorted:
                lines.append(f"[time={ev['timestamp']:.1f}s] ({ev['type']}) {ev['content'].replace(chr(10), ' ')}")
            return "\n".join(lines)

