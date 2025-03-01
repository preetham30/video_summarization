import gradio as gr
from video_summarizer import process_chat, process_screen_capture

def run_pipeline(prompt, url, modality, model_size, chunk_sec, fps):
    prompt = prompt.strip()
    url = url.strip()
    if not prompt:
        return "Error: Please enter a prompt text.", "", ""
    if not url:
        return "", "Error: Please enter a remote MP4 URL.", ""
    if not url.lower().startswith("http") or not url.lower().endswith(".mp4"):
        return "", "", "Error: URL must be an HTTP(S) link ending with .mp4"
    
    vision_text, whisper_text, merged_text = process_chat(prompt, url, modality, model_size, chunk_sec, fps)
    return vision_text, whisper_text, merged_text

# Custom CSS for red and white theme
custom_css = """
body { background-color: #fff; color: #333; font-family: Arial, sans-serif; }
h1, .gradio-title { color: #d32f2f; }
.gr-button { background-color: #d32f2f !important; color: #fff !important; }
.gradio-container { background: #fff; }
"""

with gr.Blocks(css=custom_css, title="Multimodal Video Captioning") as demo:
    gr.Markdown("# Multimodal Video Captioning")
    gr.Markdown("This ChatGPT‑like interface lets you enter a prompt and choose an input modality (Text Only, Audio Only, Vision + Audio, or All Modalities). Use the tabs below for additional functions such as screen capture processing.")
    
    with gr.Tabs():
        with gr.TabItem("Chat Interface"):
            gr.Markdown("### Chat Interface")
            modality_radio = gr.Radio(
                choices=["Text Only", "Audio Only", "Vision + Audio", "All Modalities"],
                label="Input Modality",
                value="Vision + Audio"
            )
            with gr.Row():
                prompt_box = gr.Textbox(label="Prompt Text", value="Summarize video", lines=1)
                url_box = gr.Textbox(label="Video URL or Local File Path", value="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4", lines=1)
            with gr.Row():
                whisper_model_box = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Whisper Model Size"
                )
                chunk_slider = gr.Slider(minimum=15, maximum=300, step=15, value=60, label="Audio Chunk Length (seconds)")
                fps_slider = gr.Slider(minimum=0, maximum=30, step=1, value=0, label="Frame Extraction FPS (0 to skip)")
            chat_send = gr.Button("Send")
            chat_history = gr.Chatbot(label="Chat History")
            
            def update_chat(history, prompt, vision, whisper, merged):
                history = history or []
                history.append(("User", prompt))
                history.append(("Vision Output", vision))
                history.append(("Whisper Transcript", whisper))
                history.append(("Merged Output", merged))
                return history, ""
            
            chat_send.click(
                fn=lambda prompt, url, modality, model, chunk, fps, history: update_chat(
                    history, prompt, *run_pipeline(prompt, url, modality, model, chunk, fps)
                ),
                inputs=[prompt_box, url_box, modality_radio, whisper_model_box, chunk_slider, fps_slider, chat_history],
                outputs=[chat_history, prompt_box]
            )
        
        with gr.TabItem("Screen Capture"):
            gr.Markdown("### Screen Capture Processing")
            screen_upload = gr.File(label="Upload Screen Capture Video (MP4)")
            sc_whisper_model = gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="base", label="Whisper Model Size")
            sc_chunk_slider = gr.Slider(minimum=15, maximum=300, step=15, value=60, label="Audio Chunk Length (seconds)")
            sc_fps_slider = gr.Slider(minimum=0, maximum=30, step=1, value=0, label="Frame Extraction FPS (0 to skip)")
            sc_process = gr.Button("Process Screen Capture")
            sc_vision = gr.Textbox(label="Vision Output", lines=10)
            sc_whisper = gr.Textbox(label="Whisper Transcript", lines=10)
            sc_merged = gr.Textbox(label="Merged Output", lines=10)
                    
            sc_process.click(
                fn=process_screen_capture,
                inputs=[screen_upload, sc_whisper_model, sc_chunk_slider, sc_fps_slider],
                outputs=[sc_vision, sc_whisper, sc_merged]
            )
    
demo.launch(server_port=8006, server_name="0.0.0.0", share=True)

