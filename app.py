import gradio as gr
import requests
import json
import threading

# ------------------------------------------------------
# 1) Define endpoints and model names (do not change these)
# ------------------------------------------------------
# Both text and vision tasks use the same endpoint.
QWEN_TEXT_ENDPOINT   = "http://localhost:8001/v1/chat/completions"  
QWEN_VISION_ENDPOINT = "http://localhost:8001/v1/chat/completions"  
QWEN_AUDIO_ENDPOINT  = "http://localhost:8101/v1/chat/completions"  

# Use the VL model for all text reasoning (text-text) tasks.
# This ensures that text reasoning is done by Qwen2.5-VL.
MODEL_VISION = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"         # Vision and text reasoning model
MODEL_AUDIO  = "Qwen/Qwen2.5-Audio-7B-Instruct"           # Audio transcription model

# ------------------------------------------------------
# 2) Helper functions to build payloads and POST to endpoints
# ------------------------------------------------------
def build_qwen_payload(model_name, user_content):
    """
    Construct the payload for Qwen endpoints.
    `user_content` is a list of dictionaries, for example:
      [
         {"type": "text", "text": "Describe this image"},
         {"type": "video_url", "video_url": {"url": "http://..."}}
      ]
    """
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

def post_qwen(endpoint, payload):
    """
    POST to a Qwen endpoint with the given JSON payload.
    Returns the text from the first choice if successful.
    """
    try:
        resp = requests.post(endpoint, headers={"Content-Type": "application/json"},
                             data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return "No valid response received."
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------------------------------
# 3) Define functions for each modality
# ------------------------------------------------------

# A) Text Reasoning (text-only input) using the VL model
def text_reasoning(user_text):
    if not user_text.strip():
        return "Please enter some text."
    user_content = [{"type": "text", "text": user_text}]
    # Use MODEL_VISION (Qwen2.5-VL) to process text-only queries.
    payload = build_qwen_payload(MODEL_VISION, user_content)
    return post_qwen(QWEN_VISION_ENDPOINT, payload)

# B) Vision/Video Analysis with Parallel Audio Transcription
def analyze_vision_or_video(prompt, video_input):
    if video_input is None:
        return "No video/image provided", ""
    
    # Get the local file path from the uploaded file (Gradio provides a file-like object)
    video_path = video_input.name
    results = {"vision_out": None, "audio_out": None}

    # Vision processing thread
    def call_vision():
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}}
        ]
        payload = build_qwen_payload(MODEL_VISION, user_content)
        results["vision_out"] = post_qwen(QWEN_VISION_ENDPOINT, payload)

    # Audio transcription thread
    def call_audio():
        user_content = [
            {"type": "text", "text": "Transcribe the following audio."},
            {"type": "audio_url", "audio_url": {"url": f"file://{video_path}"}}
        ]
        payload = build_qwen_payload(MODEL_AUDIO, user_content)
        results["audio_out"] = post_qwen(QWEN_AUDIO_ENDPOINT, payload)

    t1 = threading.Thread(target=call_vision)
    t2 = threading.Thread(target=call_audio)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    vision_result = results["vision_out"] or "No Vision Output"
    audio_result  = results["audio_out"] or "No Audio Output"
    return vision_result, audio_result

# C) Audio-Only Transcription
def transcribe_audio(audio_input):
    if audio_input is None:
        return "No audio provided."
    audio_path = audio_input.name
    user_content = [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}}
    ]
    payload = build_qwen_payload(MODEL_AUDIO, user_content)
    return post_qwen(QWEN_AUDIO_ENDPOINT, payload)

# D) Screen Recording Vision (extra functionality)
def analyze_screen_recording(screen_record):
    if screen_record is None:
        return "No screen recording provided."
    screen_path = screen_record.name
    user_content = [
        {"type": "text", "text": "Please describe what's happening on this screen recording."},
        {"type": "video_url", "video_url": {"url": f"file://{screen_path}"}}
    ]
    payload = build_qwen_payload(MODEL_VISION, user_content)
    return post_qwen(QWEN_VISION_ENDPOINT, payload)

# ------------------------------------------------------
# 4) Build the Gradio UI with a navigation bar (Tabs)
# ------------------------------------------------------
def build_demo():
    with gr.Blocks(title="Qwen2.5-VL Multi-Modal Front End") as demo:
        gr.Markdown("# Qwen2.5-VL Multi-Modal Front End\n\nUse the tabs below to access different processing modes. The **Home** tab retains the original video/audio processing screen.")

        with gr.Tabs():
            # Home tab: Original video and audio processing
            with gr.Tab("Home"):
                gr.Markdown("### Original Video & Audio Processing")
                home_prompt = gr.Textbox(label="Prompt", value="Describe this video or image.")
                home_video = gr.Video(label="Upload Video or Image (.mp4)")
                home_vision_out = gr.Textbox(label="Video Analysis Output", lines=6)
                home_audio_out  = gr.Textbox(label="Audio Transcription Output", lines=6)
                home_button = gr.Button("Process Video & Audio")
                home_button.click(fn=analyze_vision_or_video,
                                  inputs=[home_prompt, home_video],
                                  outputs=[home_vision_out, home_audio_out])

            # Extra: Text Reasoning Only (now using the VL model)
            with gr.Tab("Text Reasoning"):
                gr.Markdown("### Text-Based Reasoning (using Qwen2.5-VL)")
                text_input = gr.Textbox(label="Enter your text prompt")
                text_output = gr.Textbox(label="Qwen (Text) Output", lines=6)
                text_button = gr.Button("Submit")
                text_button.click(fn=text_reasoning, inputs=[text_input], outputs=[text_output])

            # Extra: Audio Transcription Only
            with gr.Tab("Audio Transcription"):
                gr.Markdown("### Audio-Only Transcription")
                audio_in = gr.Audio(label="Upload Audio (WAV, MP3, etc.)", type="filepath")
                audio_out = gr.Textbox(label="Transcribed Text")
                audio_btn = gr.Button("Transcribe")
                audio_btn.click(fn=transcribe_audio, inputs=[audio_in], outputs=[audio_out])

            # Extra: Screen Recording Vision
            with gr.Tab("Screen Recording"):
                gr.Markdown("### Screen Recording Analysis\nUpload a short video capture (representing a real-time screen recording).")
                screen_in = gr.Video(label="Upload Screen Recording (.mp4)")
                screen_out = gr.Textbox(label="Screen Recording Analysis", lines=6)
                screen_btn = gr.Button("Analyze Screen Recording")
                screen_btn.click(fn=analyze_screen_recording, inputs=[screen_in], outputs=[screen_out])

        gr.Markdown("---\n© 2025 Your Lab/Company. All Rights Reserved.")
    return demo

# ------------------------------------------------------
# 5) Launch the App
# ------------------------------------------------------
if __name__ == "__main__":
    demo = build_demo()
    # Launch on a port that is free; here we use 8003.
    demo.launch(server_name="0.0.0.0", server_port=8003,share=True)

