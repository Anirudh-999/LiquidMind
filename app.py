import requests
import json
import re

OLLAMA_API_URL = "http://localhost:11434/api/generate"

prompt = "Hello"
while True:
    prompt = input()

    if prompt == "exit":
        break

    data = {
        "model": "assistant",
        "prompt": prompt,  
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data, stream=True)
        response.raise_for_status()  

        full_response = ""  

        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded_chunk = chunk.decode('utf-8')
                    json_data = json.loads(decoded_chunk)
                    if "response" in json_data: 
                        # Clean the response to keep only the useful part
                        cleaned_response = clean_output(json_data["response"])
                        full_response += cleaned_response
                        print(cleaned_response, end="", flush=True)  
                    if json_data.get("done", False): 
                        break 
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Problematic chunk: {decoded_chunk}")  
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"Chunk content: {json_data}")

        print() 

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

# Define a function to clean the output
def clean_output(response):
    # Split the response into sentences and keep only the first part
    useful_part = response.split(".")[0]  # Keep only the first sentence
    return useful_part.strip()  # Remove any leading or trailing whitespace