import openai
import os
import json

from qdrant_client import QdrantClient
from qdrant_client.http import models


api_key = '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'

openai_client = openai.OpenAI(api_key=api_key)

# Initialize Qdrant client (replace with your Qdrant Cloud URL and API key)
client = QdrantClient(
    url="https://09932db2-aa96-47f8-a6d1-e2a4870f01ea.eu-central-1-0.aws.cloud.qdrant.io",  # e.g., "https://xyz-example.us-east.aws.cloud.qdrant.io:6333"
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQxMDkyMTQ1fQ.mODU2WVDWl0EmQYFuXNW8cnQN7B1YsH8-tzz5Bx0n88"
)

def generate_embedding(text):
    """
    Generate embeddings for the given text using OpenAI's text-embedding-3-small model.
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    
def generate_answer(query, context):
    """
    Generate an answer using OpenAI's GPT model.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None
    
def is_answer_generic(answer):
    """
    Check if the answer is generic or unhelpful.
    """
    generic_phrases = [
        "As a text-based AI",
        "I'm unable to",
        "I cannot",
        "I don't have",
        "I'm sorry",
    ]
    return any(phrase in answer for phrase in generic_phrases)

     
query_text = "How many types of filters are there?"
query_embedding = generate_embedding(query_text)
                                     
# Search in Qdrant
search_results = client.search(
    collection_name="vector_store",
    query_vector=query_embedding,
    limit=5  # Return top 5 results
)


# Display results
for result in search_results:
    print(f"Video: {result.payload['video_name']}")
    print(f"Frame: {result.payload['timestamp']}")
    print(f"Timestamp: {result.payload['summary']}")
    print(f"Timestamp: {result.payload['summary_type']}")
    print(f"Score: {result.score}")
    print("---")


confidence_threshold = 0.44  # Adjust this threshold as needed
if search_results and search_results[0].score >= confidence_threshold:
    # Extract summaries from the search results
    contexts = [result.payload["summary"] for result in search_results]

    # Generate answer
    if contexts:
        combined_context = "\n".join(contexts)
        answer = generate_answer(query_text, combined_context)
        # Check if the answer is generic or unhelpful
        if answer and not is_answer_generic(answer):
            print("Generated Answer:")
            print(answer)
        else:
            # If the answer is generic, return the timestamp of the top result
            if search_results:
                top_result = search_results[0]
                print(f"Please have a look at timestamp: {top_result.payload['timestamp']}")
            else:
                print("No relevant results found.")
    else:
        print("No relevant results found.")
else:
    # If confidence is low, return the timestamp of the top result
    if search_results:
        top_result = search_results[0]
        print(f"Please have a look at the video / timestamp: {top_result.payload['timestamp']}")
    else:
        print("No relevant results found.")

#"How many types of filters are there?"
#"How do you check oil?"
#"Can you show me a drain plug?"
#"What jeep is there in the video?"
#"Show me how to work from underneath a truck?"