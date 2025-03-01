import os
import base64
import json
import requests

# Groq API endpoint and API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Replace with your Groq API key

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to process a single image
def process_image(image_path):
    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the payload for the Groq API
    payload = {
        "model": "llama-3.2-90b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an intelligent instructional system. You must monitor and take note of all instructions related events. \
                                Start each what equipement you see and description. What action is being performed."
                             },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 1,
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    
    # Send the request to the Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    # Extract the response
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Function to process all images in a directory
def process_images_in_directory(directory_path):
    results = {}
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Process only image files
            image_path = os.path.join(directory_path, filename)
            print(f"Processing image: {filename}")
            
            # Process the image and store the result
            try:
                result = process_image(image_path)
                results[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results[filename] = {"error": str(e)}
    
    return results

# Function to save results to a JSON file
def save_results_to_json(results, output_file):
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Directory containing images
    image_directory = "extracted_frames/"  # Replace with your directory path
    
    # Output JSON file
    output_json_file = "results.json"
    
    # Process all images in the directory
    results = process_images_in_directory(image_directory)
    
    # Save results to JSON
    save_results_to_json(results, output_json_file)