import json
import requests
import os
import time
import re

# Load the final_summary.json file
def load_final_summary(json_file):
    """
    Load the final summary JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing the final summary.
    
    Returns:
        dict: Dictionary containing the final summary.
    """
    with open(json_file, "r") as f:
        return json.load(f)

# Extract valid JSON from a potentially messy response
def extract_json_from_text(text):
    """
    Extract valid JSON from text that might contain markdown, explanations, etc.
    
    Args:
        text (str): Text that might contain JSON.
    
    Returns:
        dict: Extracted JSON as a dictionary.
    """
    # Try to find JSON between triple backticks
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If not found between backticks, try to find a JSON object directly
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If all extraction attempts fail, create a default structure
    return {
        "time_segments": [
            {
                "heading": "Error in processing",
                "summary": "Could not extract valid JSON from API response. Please check the raw output."
            }
        ]
    }

# Generate a combined summary for a segment using the Groq API
def generate_combined_summary(time_segment, audio_summary, image_summaries, api_key):
    """
    Generate a combined summary for a segment using the Groq API.
    
    Args:
        time_segment (str): The time segment identifier.
        audio_summary (str): Audio summary for the segment.
        image_summaries (list): List of image summaries for the segment.
        api_key (str): Groq API key.
    
    Returns:
        dict: Combined summary in JSON format.
    """
    # Ensure image summaries are in a safe format
    image_summaries_text = "\n".join([f"- Image {i+1}: {summary}" for i, summary in enumerate(image_summaries)]) if image_summaries else "No image summaries available"

    # Optimized prompt with stronger focus on valid JSON output
    prompt = f"""
    Your task is to generate a structured summary of a video segment by combining audio and image descriptions.
    
    Time Segment: {time_segment}
    
    Audio Description:
    {audio_summary}
    
    Visual Description:
    {image_summaries_text}
    
    Create a concise summary that integrates both audio and visual information, focusing on:
    1. Top 3 Key tools and equipment shown
    2. Main actions demonstrated
    3. Important instructional points and small explanation
    
    IMPORTANT: Your response MUST be valid JSON in exactly this format:
    {{
      "time_segments": [
        {{
          "heading": "Time: {time_segment}",
          "summary": "Your integrated summary here"
        }}
      ]
    }}
    
    DO NOT include any explanations, markdown formatting, or text outside the JSON structure.
    """

    # Add a delay to prevent hitting rate limits
    time.sleep(2)

    # Prepare the API request payload
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "llama-3.2-1b-preview",
        "temperature": 0.2,  # Lower temperature for more predictable formatting
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }

    # Send the request to the Groq API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    # Extract the generated summary from the response
    if response.status_code == 200:
        try:
            # Get the raw content
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Received raw content for {time_segment}:")
            print(content[:100] + "..." if len(content) > 100 else content)
            
            # Extract JSON from potentially messy content
            return extract_json_from_text(content)
            
        except Exception as e:
            print(f"Error processing API response: {str(e)}")
            print(f"Raw content: {content}")
            # Return a default structure instead of raising an exception
            return {
                "time_segments": [
                    {
                        "heading": f"Time: {time_segment}",
                        "summary": f"Error processing segment. Audio described: {audio_summary[:100]}..."
                    }
                ]
            }
    
    elif response.status_code == 429:  # Rate limit exceeded
        retry_after = int(response.headers.get("Retry-After", 5))
        print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        return generate_combined_summary(time_segment, audio_summary, image_summaries, api_key)

    else:
        print(f"API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        # Return a default structure instead of raising an exception
        return {
            "time_segments": [
                {
                    "heading": f"Time: {time_segment}",
                    "summary": f"API request failed. Error: {response.status_code}"
                }
            ]
        }

# Process each segment and generate combined summaries
def process_segments(final_summary, api_key):
    """
    Process each time segment and generate combined summaries using the Groq API.
    
    Args:
        final_summary (dict): Dictionary containing the final summary.
        api_key (str): Groq API key.
    
    Returns:
        dict: Dictionary with combined summaries for each segment.
    """
    combined_summaries = {"time_segments": []}

    for time_segment, summaries in final_summary.items():
        print(f"\nProcessing time segment: {time_segment}")
        
        # Extract audio and image summaries
        audio_summary = summaries.get("Audio Summary", "No audio summary available")
        image_summaries = summaries.get("Image Summaries", [])

        try:
            # Generate combined summary using the Groq API
            combined_summary = generate_combined_summary(time_segment, audio_summary, image_summaries, api_key)

            # Append the combined summary to the result
            if "time_segments" in combined_summary and combined_summary["time_segments"]:
                combined_summaries["time_segments"].append(combined_summary["time_segments"][0])
                print(f"Successfully added summary for {time_segment}")
            else:
                # Handle unexpected structure
                print(f"Warning: Unexpected structure in API response for {time_segment}")
                combined_summaries["time_segments"].append({
                    "heading": f"Time: {time_segment}",
                    "summary": "Error: Unexpected structure in API response"
                })
        except Exception as e:
            print(f"Error processing segment {time_segment}: {str(e)}")
            # Add an error entry to maintain continuity
            combined_summaries["time_segments"].append({
                "heading": f"Time: {time_segment}",
                "summary": f"Error: {str(e)}"
            })
            
    return combined_summaries

# Save the combined summaries to a JSON file
def save_combined_summaries(combined_summaries, output_file):
    """
    Save the combined summaries to a JSON file.
    
    Args:
        combined_summaries (dict): Dictionary containing the combined summaries.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(combined_summaries, f, indent=4)
    print(f"Combined summaries saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Paths and API key
    final_summary_file = "final_summary.json"  # Replace with your final summary file path
    output_file = "combined_summaries.json"  # Output file for the combined summaries
    
    # Get API key from environment variable for security (recommended method)
    groq_api_key = "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"
    

    # Ensure API key is available
    if not groq_api_key or groq_api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Missing Groq API key. Set the GROQ_API_KEY environment variable.")

    try:
        # Load the final summary
        final_summary = load_final_summary(final_summary_file)
        
        # Process each segment and generate combined summaries
        combined_summaries = process_segments(final_summary, groq_api_key)
        
        # Save the combined summaries to a JSON file
        save_combined_summaries(combined_summaries, output_file)
        
        print("\nSummary generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()