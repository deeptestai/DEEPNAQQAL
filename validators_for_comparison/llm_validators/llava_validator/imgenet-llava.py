import os
import base64
import requests
import csv
import time

# Configuration
BASE_FOLDER = "imagenet_png"  # Folder containing all images
SUBFOLDERS = ["imagenet_dlf", "imagenet_dx", "imagenet_ox", "imagenet_sv"]  # Subfolders
LLM_API_URL = "http://localhost:11434/v1/chat/completions"  # LLaMA Vision API
CSV_FILE = "img_llama_run2.csv"  # Output file
MODEL_NAME = "llava"  # Updated model name

# Function to encode an image to Base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

# Function to send image to LLaMA 3.2 Vision for classification
def classify_image(image_base64):
    prompt = (
        "Imagine you are an expert image assessor. "
        "You have to provide an evaluation of validity based on this definition: "
        "'Valid means recognizable by humans as part of the input domain, e.g., if an image of pizza "
        "belongs to the domain of real-world objects'.\n\n"
        "I provide you an image representing a pizza. If the image actually represents a real-world object, "
        "you answer 'id'. If the image does not represent a real-world object, you answer 'ood'.\n\n"
        "Let's start the task. I provide you folder of images. "
        "You evaluate each single PNG image. "
        "Return only 'id' or 'ood'. Do not include any other text or explanations."
    )


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

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Request failed: {e}")
        return "ood"

# Process all images in subfolders
def process_images():
    results = []

    for subfolder in SUBFOLDERS:
        folder_path = os.path.join(BASE_FOLDER, subfolder)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        print(f"Processing folder: {subfolder}")

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                image_path = os.path.join(folder_path, file_name)
                image_base64 = encode_image(image_path)
                classification = classify_image(image_base64)

                results.append([file_name, subfolder, classification])
                print(f"Processed {file_name} → Class: {classification}")

                time.sleep(0.5)  # Prevent API overload

    # Save results to CSV
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Folder", "Classification"])
        writer.writerows(results)

    print(f"\nClassification complete! Results saved in {CSV_FILE}")

# Run the script
if __name__ == "__main__":
    process_images()
