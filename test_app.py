import gradio as gr

# ------------------------------------------------------
# 1) Define the function to handle camera recording
# ------------------------------------------------------
def handle_camera_recording(video_input):
    """
    Handle the camera recording input.
    This function simply returns the video for display.
    """
    if video_input is None:
        return "No video provided."
    return video_input

# ------------------------------------------------------
# 2) Build the Gradio UI
# ------------------------------------------------------
def build_demo():
    with gr.Blocks(title="Camera Recording App") as demo:
        gr.Markdown("# Camera Recording App\n\nUse the webcam to record a video and display it on the screen.")

        # Camera recording section
        with gr.Row():
            camera_input = gr.Video(label="Upload Video")  # Upload a video file (No direct webcam support)
            camera_output = gr.Video(label="Recorded Video")  # Display the recorded video

        # Button to process the recorded video
        record_button = gr.Button("Upload and Display")
        record_button.click(fn=handle_camera_recording,
                            inputs=[camera_input],
                            outputs=[camera_output])

        gr.Markdown("---\n© 2025 Your Lab/Company. All Rights Reserved.")
    return demo

# ------------------------------------------------------
# 3) Launch the App
# ------------------------------------------------------
if __name__ == "__main__":
    demo = build_demo()
    # Launch on a port that is free; here we use 8003.
    demo.launch(server_name="0.0.0.0", server_port=8003, share=True)
