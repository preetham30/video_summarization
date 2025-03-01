import json
import requests
import os
import time
import re

groq_api_key = "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"

def load_final_summary(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def extract_json_from_text(text):
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return {
        "time_segments": [
            {
                "heading": "Error in processing",
                "summary": "Could not extract valid JSON."
            }
        ]
    }

def generate_combined_summary(time_segment, audio_summary, image_summaries, api_key):
    image_summaries_text = "\n".join([f"- Image {i+1}: {s}" for i, s in enumerate(image_summaries)]) \
        if image_summaries else "No image summaries"
    prompt = f"""
    Your task is to generate a structured summary...
    Time Segment: {time_segment}

    Audio Description:
    {audio_summary}

    Visual Description:
    {image_summaries_text}

    IMPORTANT: Must be valid JSON in exactly this format:
    {{
      "time_segments": [
        {{
          "heading": "Time: {time_segment}",
          "summary": "Your integrated summary"
        }}
      ]
    }}
    """
    time.sleep(2)
    payload = {
        "messages": [
            {"role":"user","content": prompt}
        ],
        "model": "llama-3.2-1b-preview",
        "temperature": 0.2,
        "max_completion_tokens":1024,
        "top_p":1,
        "stream":False,
        "stop":None
    }
    headers = {
        "Content-Type":"application/json",
        "Authorization": f"Bearer {api_key}"
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    if r.status_code==200:
        try:
            content = r.json()["choices"][0]["message"]["content"]
            return extract_json_from_text(content)
        except Exception as e:
            return {
                "time_segments": [
                    {
                        "heading": f"Time: {time_segment}",
                        "summary": f"Error parsing: {e}"
                    }
                ]
            }
    elif r.status_code==429:
        retry_after = int(r.headers.get("Retry-After",5))
        time.sleep(retry_after)
        return generate_combined_summary(time_segment,audio_summary,image_summaries,api_key)
    else:
        return {
            "time_segments": [
                {
                    "heading": f"Time: {time_segment}",
                    "summary": f"API failed with {r.status_code}"
                }
            ]
        }

def process_segments(final_summary, api_key):
    combined_summaries = {"time_segments":[]}
    for time_segment, summaries in final_summary.items():
        audio_summary = summaries.get("Audio Summary","No audio summary")
        image_summaries = summaries.get("Image Summaries",[])
        try:
            combined = generate_combined_summary(time_segment,audio_summary,image_summaries,api_key)
            if "time_segments" in combined and combined["time_segments"]:
                combined_summaries["time_segments"].append(combined["time_segments"][0])
        except Exception as e:
            combined_summaries["time_segments"].append({
                "heading": f"Time: {time_segment}",
                "summary": f"Error: {e}"
            })
    return combined_summaries

def save_combined_summaries(combined_summaries, output_file):
    with open(output_file,"w") as f:
        json.dump(combined_summaries,f,indent=4)
    print(f"Combined summaries saved to {output_file}")

if __name__=="__main__":
    final_summary_file = "final_summary.json"
    output_file = "combined_summaries.json"
    if not groq_api_key:
        raise ValueError("Missing groq_api_key.")
    final_summary = load_final_summary(final_summary_file)
    combined_summaries = process_segments(final_summary, groq_api_key)
    save_combined_summaries(combined_summaries, output_file)
    print("Summary generation complete.")

