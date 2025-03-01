import os
import base64
import json
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_image(image_path):
    base64_image = encode_image_to_base64(image_path)
    payload = {
        "model": "llama-3.2-90b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an intelligent system..."
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
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed with {response.status_code}: {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"]

def process_images_in_directory(directory_path):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".jpg",".jpeg",".png")):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing image: {filename}")
            try:
                desc = process_image(image_path)
                results[filename] = desc
            except Exception as e:
                print(f"Error on {filename}: {e}")
                results[filename] = {"error": str(e)}
    return results

def save_results_to_json(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__=="__main__":
    image_directory = "extracted_frames"
    output_json_file = "results.json"
    results = process_images_in_directory(image_directory)
    save_results_to_json(results, output_json_file)

