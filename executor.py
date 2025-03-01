from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import os
import json
import openai

import hashlib
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cv2
from moviepy.editor import VideoFileClip
import whisper
from pydub import AudioSegment
from run_vision_model import process_image
from embedding_generator import generate_embedding

print('Imports are good.')

# Step 1: Frame and Audio Extraction
def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: No frames found. Invalid or corrupt video file.")
        return
    print(f"Extracting 1 frame every {frame_interval} frames.")
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
    cap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")

def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from a video.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    print(f"Audio extracted and saved to {output_audio_path}")

# Step 2: Audio Transcription (using Whisper)
def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Step 3: Image Analysis (using Groq)
def analyze_images(image_dir):
    """
    Analyzes images in a directory using Groq.
    """
    results = {}
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing image: {filename}")
            try:
                result = process_image(image_path)  # Your existing Groq image processing function
                results[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results[filename] = {"error": str(e)}

    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to results.json")

    return results

# Step 4: Alignment of Audio and Image Summaries
def load_image_summaries(json_file):
    """
    Load image summaries from a JSON file.
    """
    with open(json_file, "r") as f:
        return json.load(f)

def load_audio_summaries(json_file):
    """
    Load audio summaries from a JSON file containing transcriptions.
    """
    with open(json_file, "r") as f:
        transcriptions = json.load(f)
    audio_summaries = {}
    for i, entry in enumerate(transcriptions, start=1):
        audio_summaries[i] = entry["text"]  # Use chunk ID (i) as the key
    return audio_summaries

def assign_image_timestamps(image_summaries, frame_rate=30):
    """
    Assign timestamps to image summaries based on frame numbers and frame rate.
    """
    image_timestamps = {}
    for frame_name, summary in image_summaries.items():
        frame_number = int(frame_name.split("_")[1].split(".")[0]) * 120
        timestamp = frame_number / frame_rate  # Time in seconds
        image_timestamps[timestamp] = summary
    return image_timestamps

def align_summaries(audio_summaries, image_timestamps, chunk_duration=15):
    """
    Align audio and image summaries by time.
    """
    aligned_summaries = {}
    for chunk_number, audio_summary in audio_summaries.items():
        if chunk_number < 1:
            continue  # Skip invalid chunk numbers
        start_time = (chunk_number - 1) * chunk_duration
        end_time = chunk_number * chunk_duration
        image_summaries_in_chunk = []
        for timestamp, image_summary in image_timestamps.items():
            if start_time <= timestamp < end_time:
                image_summaries_in_chunk.append(image_summary)
        aligned_summaries[f"{start_time}-{end_time} sec"] = {
            "audio_summary": audio_summary,
            "image_summaries": image_summaries_in_chunk,
        }
    return aligned_summaries

def generate_final_summary(aligned_summaries):
    """
    Generate a final summary for each time segment by combining audio and image summaries.
    """
    final_summary = {}
    for time_segment, summaries in aligned_summaries.items():
        audio_summary = summaries["audio_summary"]
        image_summaries = summaries["image_summaries"]
        final_summary[time_segment] = {
            "Audio Summary": audio_summary,
            "Image Summaries": image_summaries
        }
    return final_summary

# Step 5: Summarization (using LangChain and Groq)
def summarize_segment(time_segment, audio_summary, image_summaries, groq_model):
    """
    Summarizes a video segment using LangChain and Groq.
    """
    prompt = PromptTemplate(
        input_variables=["time_segment", "audio_summary", "image_summaries"],
        template="""
        Your task is to generate a structured summary of a video segment by combining audio and image descriptions.
        
        Time Segment: {time_segment}
        
        Audio Description:
        {audio_summary}
        
        Visual Description:
        {image_summaries}
        
        Create a concise summary that integrates both audio and visual information, focusing on:
        1. Top 3 Key tools and equipment shown
        2. Main actions demonstrated
        3. Important instructional points and small explanation
        """
    )
    chain = LLMChain(llm=groq_model, prompt=prompt)
    return chain.run({
        "time_segment": time_segment,
        "audio_summary": audio_summary,
        "image_summaries": "\n".join(image_summaries) if image_summaries else "No image summaries available"
    })

# Step 6: Insert Summaries into Qdrant
def insert_to_qdrant(video_name, time_segment, audio_summary, image_summaries, openai_api_key, qdrant_client):
    """
    Inserts summaries into Qdrant.
    """
    audio_embedding = generate_embedding(audio_summary)
    if audio_embedding:
        point_id = generate_id(video_name, time_segment, 'audio')
        qdrant_client.upsert(
            collection_name="vector_store",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=audio_embedding,
                    payload={
                        "video_name": video_name,
                        "timestamp": time_segment,
                        "summary_type": "audio",
                        "summary": audio_summary
                    }
                )
            ]
        )
        print(f"Inserted audio summary for segment: {time_segment}")
    else:
        print(f"Failed to generate embedding for audio summary: {time_segment}")

    for i, image_summary in enumerate(image_summaries):
        image_embedding = generate_embedding(image_summary)
        if image_embedding:
            point_id = generate_id(video_name, time_segment, f'image_{i}')
            qdrant_client.upsert(
                collection_name="vector_store",
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=image_embedding,
                        payload={
                            "video_name": video_name,
                            "timestamp": time_segment,
                            "summary_type": "image",
                            "summary": image_summary
                        }
                    )
                ]
            )
            print(f"Inserted image summary {i} for segment: {time_segment}")
        else:
            print(f"Failed to generate embedding for image summary {i}: {time_segment}")

# Main Orchestration Function
def orchestrate_pipeline(video_path, output_folder, openai_api_key, groq_api_key, qdrant_url, qdrant_api_key):
    """
    Orchestrates the entire video summarization pipeline.
    """
    # Step 1: Extract frames and audio
    extract_frames(video_path, output_folder, frame_interval=120)
    extract_audio(video_path, os.path.join(output_folder, "extracted_audio.mp3"))

    # Step 2: Transcribe audio
    audio_transcription = transcribe_audio(os.path.join(output_folder, "extracted_audio.mp3"))

    # Step 3: Analyze images
    image_results = analyze_images(output_folder)

    # Step 4: Align audio and image summaries
    image_summaries = load_image_summaries("results.json")
    audio_summaries = load_audio_summaries(output_folder+"transcriptions.json")
    image_timestamps = assign_image_timestamps(image_summaries)
    aligned_summaries = align_summaries(audio_summaries, image_timestamps)
    final_summary = generate_final_summary(aligned_summaries)

    # Step 5: Summarize segments
    groq_model = ChatGroq(api_key=groq_api_key, model="llama-3.2-1b-preview")
    for time_segment, summaries in final_summary.items():
        audio_summary = summaries["Audio Summary"]
        image_summaries = summaries["Image Summaries"]
        combined_summary = summarize_segment(time_segment, audio_summary, image_summaries, groq_model)
        final_summary[time_segment]["Combined Summary"] = combined_summary

    # Step 6: Insert summaries into Qdrant
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    video_name = os.path.basename(video_path)
    for time_segment, summaries in final_summary.items():
        insert_to_qdrant(
            video_name,
            time_segment,
            summaries["Audio Summary"],
            summaries["Image Summaries"],
            openai_api_key,
            qdrant_client
        )

    print("Pipeline completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    # Configuration
    video_path = "sample_video_3.mp4"  # Replace with your video path
    output_folder = "extracted_frames/"  # Folder for extracted frames and audio
    openai_api_key =  os.getenv("OPENAI_API_KEY") # Replace with your OpenAI API key
    groq_api_key = "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"  # Replace with your Groq API key
    qdrant_url = "https://09932db2-aa96-47f8-a6d1-e2a4870f01ea.eu-central-1-0.aws.cloud.qdrant.io"  # Replace with your Qdrant URL
    qdrant_api_key =  os.getenv("QDRANT_API_KEY") # Replace with your Qdrant API key

    # Run the orchestration
    orchestrate_pipeline(video_path, output_folder, openai_api_key, groq_api_key, qdrant_url, qdrant_api_key)