import streamlit as st
import json
import re

# Sample video files and their corresponding JSON summary files
video_files = {
    "sample_video_3.mp4": "combined_summaries.json"
}

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

# Function to display video and summary
def display_video_and_summary(selected_video):
    col1, col2 = st.columns(2)
    
    with col1:
        st.video(selected_video)
    
    with col2:
        st.write("**Summary:**")
        summary_data = read_summary(video_files[selected_video])
        
        if summary_data:
            for item in summary_data:
                heading = item.get("heading", "No heading available")
                summary = item.get("summary", "No summary available")
                
                # Extract the first digit from the heading (e.g., "Time: 0-60 sec" -> 0)
                timestamp = extract_first_digit(heading)
                
                # Display the heading, summary, and a clickable link
                st.markdown(f"**{heading}**: {summary} [Go to {timestamp}s](#{timestamp})", unsafe_allow_html=True)
        else:
            st.warning("No summary data available.")

# Function to handle query
def handle_query(query):
    st.write(f"**Query Result:** You entered: {query}")

# Streamlit app
def main():
    st.title("Video Summary and Query App")
    
    # Dropdown to select video
    selected_video = st.selectbox("Select a video", list(video_files.keys()))
    
    # Display video and summary
    display_video_and_summary(selected_video)
    
    # Query section
    st.write("### Query Section")
    query = st.text_input("Enter your query:")
    
    if st.button("Run"):
        if query:
            handle_query(query)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()