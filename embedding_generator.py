import openai
import os
import json
import random
import hashlib
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize OpenAI Client
openai_client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY")  )
qdrant_url = "https://09932db2-aa96-47f8-a6d1-e2a4870f01ea.eu-central-1-0.aws.cloud.qdrant.io"  # Replace with your Qdrant URL
qdrant_api_key = os.getenv("QDRANT_API_KEY") # Replace with your Qdrant API key

# Initialize Qdrant Client
client = QdrantClient(
    url=qdrant_url,  # Store API URL as environment variable
    api_key=qdrant_api_key
)

# Function to Generate Unique Point IDs
def generate_id(video_name, heading, summary_type):
    unique_string = f"{video_name}_{heading}_{summary_type}"
    return int(hashlib.sha256(unique_string.encode()).hexdigest(), 16) % 10**18 + random.randint(0, 99999)

# Function to Generate OpenAI Embeddings
def generate_embedding(text):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]  # Ensure input is a list
        )
        return np.array(response.data[0].embedding, dtype=np.float32).tolist()
    except Exception as e:
        print(f" Error generating embedding: {e}")
        return None

# Function to Convert Time-Stamps
def transform_timestamp(heading):
    start_time = heading.split("-")[0].replace("Time: ", "").replace("s", "")
    if int(start_time) < 60:
        return f"{start_time}s"
    minutes = int(start_time) // 60
    seconds = int(start_time) % 60
    return f"{minutes}:{seconds:02d}"

# Function to Ingest Summaries into Qdrant
def ingest_summaries(json_file, video_name="sample_video_1.mp4"):
    with open(json_file, "r") as file:
        data = json.load(file)

    for segment, summaries in data.items():
        transformed_timestamp = transform_timestamp(segment)
        audio_summary = summaries.get("Audio Summary", "")

        # Insert Audio Summary
        audio_embedding = generate_embedding(audio_summary)
        if audio_embedding:
            point_id = generate_id(video_name, transformed_timestamp, 'audio')
            client.upsert(
                collection_name="vector_store",
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=audio_embedding,
                        payload={
                            "video_name": video_name,
                            "timestamp": transformed_timestamp,
                            "summary_type": "audio",
                            "summary": audio_summary
                        }
                    )
                ]
            )
            print(f"Inserted Audio Summary for {transformed_timestamp}")
        else:
            print(f"Failed to generate embedding for audio summary: {transformed_timestamp}")

        # Insert Image Summaries
        for i, image_summary in enumerate(summaries.get("Image Summaries", [])):
            image_embedding = generate_embedding(image_summary)
            if image_embedding:
                point_id = generate_id(video_name, transformed_timestamp, f'image_{i}')
                client.upsert(
                    collection_name="vector_store",
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=image_embedding,
                            payload={
                                "video_name": video_name,
                                "timestamp": transformed_timestamp,
                                "summary_type": "image",
                                "summary": image_summary
                            }
                        )
                    ]
                )
                print(f"Inserted Image Summary {i} for {transformed_timestamp}")
            else:
                print(f"Failed to generate embedding for image summary {i}: {transformed_timestamp}")

    print("🎉 Data Ingestion to Qdrant Completed!")

# Run `main()` only if executed directly (not imported)
if __name__ == "__main__":
    ingest_summaries("final_summary.json")
