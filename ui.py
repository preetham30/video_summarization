import streamlit as st
import requests
import os
import uuid
import json

# Local modules from your codebase
import frame_extractor
import audio_transcription
import align
import summarize

# HPC endpoint for Qwen
QWEN_API_URL = "http://localhost:8001/v1/chat/completions"  # Adjust if needed

def run_local_pipeline(video_path, chunk_ms=60000):
    """
    Runs your local pipeline:
      1) Extract frames from `video_path`
      2) Run run_vision_model.py -> Summaries => results.json
      3) Extract audio => transcribe in chunks => transcriptions.json
      4) Align with images => final_summary.json
      5) Summarize => combined_summaries.json
    Returns the path to 'combined_summaries.json'.
    """
    import subprocess
    import shutil

    # 1) Extract frames
    frames_folder = "extracted_frames"
    st.write("Extracting frames...")
    if os.path.isdir(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder, exist_ok=True)

    frame_extractor.extract_frames(
        video_path=video_path,
        output_folder=frames_folder,
        frame_interval=120  # example: 1 frame every 120 frames
    )

    st.write("Frames extracted successfully.")

    # 2) run_vision_model.py => results.json
    st.write("Running Qwen-based vision model (run_vision_model.py) on extracted frames...")
    # We can do it via a subprocess or a direct function call:
    #   "python run_vision_model.py" => but run_vision_model has a main check.
    #   For demonstration, we do a direct import trick:
    import run_vision_model
    run_vision_model.image_directory = frames_folder
    run_vision_model.output_json_file = "results.json"
    results_out = "results.json"
    # But we see run_vision_model.py uses "process_images_in_directory(image_directory)"...
    # Let's just call that function:
    results = run_vision_model.process_images_in_directory(frames_folder)
    run_vision_model.save_results_to_json(results, results_out)

    st.write(f"Image results saved to {results_out}.")

    # 3) Extract audio => "extracted_audio.mp3"
    st.write("Extracting audio from video...")
    audio_path = "extracted_audio.mp3"
    frame_extractor.extract_audio(video_path, audio_path)

    st.write(f"Audio saved to {audio_path}.")

    # 4) Transcribe in chunks => transcriptions/transcriptions.json
    st.write("Transcribing audio in chunks with Whisper...")
    # example
    import audio_transcription
    model = audio_transcription.load_whisper_model("tiny")
    trans_dir = "transcriptions"
    if os.path.isdir(trans_dir):
        shutil.rmtree(trans_dir)
    os.makedirs(trans_dir, exist_ok=True)

    transcriptions = audio_transcription.process_audio_in_chunks(
        model, audio_path, chunk_length_ms=chunk_ms, output_dir=trans_dir
    )
    # This produces "transcriptions.json" inside trans_dir
    # But your align.py expects a single "transcriptions.json" in root?
    # Let's move it:
    src_json = os.path.join(trans_dir, "transcriptions.json")
    final_trans_json = "transcriptions.json"
    os.rename(src_json, final_trans_json)

    st.write(f"Audio transcriptions => {final_trans_json}")

    # 5) Align => final_summary.json
    st.write("Aligning image and audio summaries (align.py)...")
    import align
    # align uses 'results.json' + 'transcriptions.json'
    # final output => final_summary.json
    # We'll do the same steps as align's main
    image_summaries = align.load_image_summaries("results.json")
    audio_summaries = align.load_audio_summaries("transcriptions.json")
    image_timestamps = align.assign_image_timestamps(image_summaries)
    aligned_summaries = align.align_summaries(audio_summaries, image_timestamps)
    final_summary = align.generate_final_summary(aligned_summaries)
    align.save_final_summary(final_summary, "final_summary.json")

    st.write("Created final_summary.json")

    # 6) Summarize => combined_summaries.json
    st.write("Using summarize.py to further combine segments (Groq calls, etc.)")
    import summarize
    # This will read final_summary.json => produce combined_summaries.json
    # HPC calls to Groq or Qwen might happen here. We'll just do the main:
    summarize.final_summary_file = "final_summary.json"
    summarize.output_file = "combined_summaries.json"
    # We'll do a direct approach:
    try:
        # The code in summarize.py's main is:
        final_summary_data = summarize.load_final_summary("final_summary.json")
        combined_summaries = summarize.process_segments(final_summary_data, summarize.groq_api_key)
        summarize.save_combined_summaries(combined_summaries, "combined_summaries.json")
    except Exception as e:
        st.write(f"Error in summarize step: {e}")

    st.write("Local pipeline done. Final: combined_summaries.json")
    return "combined_summaries.json"


def post_describe_video(video_url):
    """
    POST to HPC Qwen with the given video_url, as your cURL example:
    {
      "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type":"text","text":"Describe this video."},
            {"type":"video_url","video_url":{"url":"<video_url>"}}
          ]
        }
      ]
    }
    """
    payload = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type":"text","text":"Describe this video."},
                    {"type":"video_url","video_url":{"url": video_url}}
                ]
            }
        ]
    }
    headers = {"Content-Type":"application/json"}

    try:
        resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            return f"Error {resp.status_code} from HPC: {resp.text}"
        data = resp.json()
        if "choices" in data and len(data["choices"])>0:
            return data["choices"][0]["message"].get("content","No content found")
        else:
            return "No valid response from HPC."
    except Exception as e:
        return f"Exception calling HPC: {e}"


def main():
    st.title("Unified Video Summarization & Qwen2.5-VL Demo")

    st.write("This demo has two tabs:")
    tab = st.tabs(["Local Summarization Pipeline","Describe with Qwen"])

    # Tab 1: Local Summarization Pipeline
    with tab[0]:
        st.header("Local Summarization Pipeline")
        st.write("1) Provide a local video path on server. (No file upload in this example.)")
        video_path = st.text_input("Local video path (e.g. 'sample_video_3.mp4'):", "")
        chunk_len = st.number_input("Audio chunk length (seconds)", min_value=10, value=60)
        if st.button("Run Local Pipeline"):
            if not video_path.strip():
                st.warning("Please enter a local video path.")
            else:
                result_json = run_local_pipeline(video_path.strip(), chunk_ms=chunk_len*1000)
                st.success(f"Pipeline finished. Final summary => {result_json}")
                # Optionally display final text:
                if os.path.exists(result_json):
                    with open(result_json,"r") as f:
                        data = json.load(f)
                    st.json(data)

    # Tab 2: HPC Qwen describe
    with tab[1]:
        st.header("Describe with Qwen2.5-VL HPC")
        st.write("Sends a POST to HPC with 'video_url' + 'Describe this video.'")
        user_video_url = st.text_input("HTTP video URL:", "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4")
        if st.button("Describe with HPC"):
            if not user_video_url.strip():
                st.warning("Please provide a valid HTTP video URL.")
            else:
                answer = post_describe_video(user_video_url.strip())
                st.write("**HPC Response:**")
                st.write(answer)


if __name__=="__main__":
    main()

