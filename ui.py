import streamlit as st
import json
import re
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Sample video files and their corresponding JSON summary files
video_files = {
    "sample_video_1.mp4": "combined_summaries_1.json",
    "sample_video_2.mp4": "combined_summaries_2.json"
}

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=  os.getenv("OPENAI_API_KEY"))  # Replace with your OpenAI API key

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://09932db2-aa96-47f8-a6d1-e2a4870f01ea.eu-central-1-0.aws.cloud.qdrant.io",  # Replace with your Qdrant URL
    api_key= os.getenv("QDRANT_API_KEY")  # Replace with your Qdrant API key
)

# Function to read summary from JSON file
def read_summary(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            # Check if the JSON has the expected structure
            if isinstance(data, dict) and "time_segments" in data:
                return data["time_segments"]
            else:
                st.error(f"Invalid JSON format in {file_path}. Expected a dictionary with 'time_segments' key.")
                return []
    except Exception as e:  
        st.error(f"Error reading {file_path}: {e}")
        return []

# Function to extract the first digit from a string
def extract_first_digit(time_string):
    match = re.search(r"\d+", time_string)  # Find the first sequence of digits
    if match:
        return int(match.group())  # Convert to integer
    return 0  # Default to 0 if no digits are found

# Function to generate embeddings using OpenAI
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
        st.error(f"Error generating embedding: {e}")
        return None

# Function to generate an answer using OpenAI's GPT model
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
        st.error(f"Error generating answer: {e}")
        return None

# Function to check if the answer is generic or unhelpful
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

# Function to handle query
def handle_query(query, selected_video):
    """
    Handle user query by searching in Qdrant and generating an answer using OpenAI.
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    if not query_embedding:
        st.error("Failed to generate embedding for the query.")
        return

    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name="vector_store",
        query_vector=query_embedding,
        limit=5  # Return top 5 results
    )

    # Display results
    if search_results:
        confidence_threshold = 0.44  # Adjust this threshold as needed
        if search_results[0].score >= confidence_threshold:
            # Extract summaries from the search results
            contexts = [result.payload["summary"] for result in search_results]

            # Generate answer
            if contexts:
                combined_context = "\n".join(contexts)
                answer = generate_answer(query, combined_context)
                # Check if the answer is generic or unhelpful
                if answer and not is_answer_generic(answer):
                    st.write("**Generated Answer:**")
                    st.write(answer)
                else:
                    # If the answer is generic, return the timestamp of the top result
                    top_result = search_results[0]
                    st.write(f"Please have a look at timestamp: {top_result.payload['timestamp']}")
            else:
                st.warning("No relevant results found.")
        else:
            # If confidence is low, return the timestamp of the top result
            top_result = search_results[0]
            st.write(f"Please have a look at the video / timestamp: {top_result.payload['timestamp']}")
    else:
        st.warning("No relevant results found.")

# Function to display video and summary
def display_video_and_summary(selected_video):
    # Initialize session state for video start time
    if "start_time" not in st.session_state:
        st.session_state.start_time = 0

    # Display the video with the current start time
    st.video(selected_video, format="video/mp4", start_time=st.session_state.start_time)

    # Display the summary in a scrollable modal
    st.write("**Summary:**")
    summary_data = read_summary(video_files[selected_video])
    
    if summary_data:
        # Create a scrollable container for the summary
        with st.expander("View Summary", expanded=True):
            # Add custom CSS to make the content scrollable
            st.markdown(
                """
                <style>
                .scrollable {
                    max-height: 300px;
                    overflow-y: auto;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Wrap the summary content in a scrollable div
            st.markdown('<div class="scrollable">', unsafe_allow_html=True)
            for idx, item in enumerate(summary_data):  # Use enumerate to get a unique index
                heading = item.get("heading", "No heading available")
                summary = item.get("summary", "No summary available")
                
                # Extract the first digit from the heading (e.g., "Time: 0-60 sec" -> 0)
                timestamp = extract_first_digit(heading)
                
                # Create a button to navigate to the timestamp
                if st.button(f"Go to {timestamp}s", key=f"button_{timestamp}_{idx}"):  # Include index in the key
                    st.session_state.start_time = timestamp
                    st.experimental_rerun()  # Rerun the app to update the video start time
                
                # Display the heading and summary
                st.markdown(f"**{heading}**: {summary}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No summary data available.")

# Streamlit app
def main():
    st.title("VidBot🤖")
    
    # Dropdown to select video
    selected_video = st.selectbox("Select a video", list(video_files.keys()))
    
    # Display video and summary
    display_video_and_summary(selected_video)
    
    # Query section
    st.write("### Query Section")
    query = st.text_input("Enter your query:")
    
    if st.button("Run"):
        if query:
            handle_query(query, selected_video)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()