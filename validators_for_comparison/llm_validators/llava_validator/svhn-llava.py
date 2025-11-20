import os
import base64
import requests
import csv

# Configuration
BASE_FOLDER = "output_images"  # Folder containing all images
SUBFOLDERS = ["svhn_dj","svhn_dlf", "svhn_dx", "svhn_ox", "svhn_sv"]  # Subfolders
LLM_API_URL = "http://localhost:11434/v1/chat/completions"  # LLaMA Vision API
CSV_FILE = "svhn_llava_run2.csv"  # Output file
MODEL_NAME = "llava"  # Updated model name

# Function to encode an image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to send image to LLaMA 3.2 Vision for classification
def classify_image(image_base64):
    prompt = (
        "Imagine you are an expert image assessor. "
        "You have to provide an evaluation of validity based on this definition: "
        "'Valid means recognizable by humans as part of the input domain, "
        "e.g., if an image of house number belongs to the domain of house number'.\n\n"
        "I provide you urls of images. If the image actually represents an image from the domain of house number,you answer 'id'."
        " If the image does not represent a house number, you answer 'ood'.\n\n"
        "Return only 'id' or 'ood'. Do not include any other text, or explanations."
        )

    # API Payload
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Evaluate this image based on the definition provided."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]
    }

    headers = {"Content-Type": "application/json"}
    
    # Send the request
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        try:
            # Extract classification response (expected 'id' or 'ood')
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            return result_text  # Should return 'id' or 'ood'
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "ood"  # Default to 'ood' if an error occurs
    return "ood"  # Default classification for failed requests

# Process all images in the main folder
def process_images():
    results = []

    if not os.path.exists(BASE_FOLDER):
        print(f"Folder not found: {BASE_FOLDER}")
        return

    for file_name in os.listdir(BASE_FOLDER):
        if file_name.endswith(".png"):
            image_path = os.path.join(BASE_FOLDER, file_name)
            image_base64 = encode_image(image_path)
            classification = classify_image(image_base64)
            
            results.append([file_name, classification])
            print(f"Processed {file_name} → Class: {classification}")

    # Save results to CSV
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Classification"])
        writer.writerows(results)

    print(f"\nClassification complete! Results saved in {CSV_FILE}")

# Run the script
if __name__ == "__main__":
    process_images()
