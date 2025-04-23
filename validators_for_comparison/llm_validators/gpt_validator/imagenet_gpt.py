import openai
import pandas as pd
import os
# Print API key from environment
print("Environment Key:", os.getenv("OPENAI_API_KEY"))
# Set API key explicitly for OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to your CSV file (Modify this path)
CSV_FILE_PATH = "img_link.csv"
df = pd.read_csv(CSV_FILE_PATH, header=None)
print(df.head(10))

# Output file
OUTPUT_CSV = "img_classified_images.csv"

# Load the CSV file
df = pd.read_csv(CSV_FILE_PATH, header=None)

# Flatten the DataFrame (since all URLs are in a single row)
image_urls = df.values.flatten()

# Filter out unwanted images (do not classify vr_0_e281_pNone.png)
filtered_urls = [url for url in image_urls if "vr_0_e281_pNone.png" not in url]

# Store results
results = []

# Iterate through each image URL
for image_url in filtered_urls:
    try:
        # Send image to GPT-4o using the new format
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": (
                    "Imagine you are an expert image assessor. "
                    "You have to provide an evaluation of validity based on this definition: "
                    "'Valid means recognizable by humans as part of the input domain, e.g., if an image of pizza "
                    "belongs to the domain of real-world objects'.\n\n"
                    "I provide you an image representing a pizza. If the image actually represents a real-world object, "
                    "you answer 'id'. If the image does not represent a real-world object, you answer 'ood'.\n\n"
                    "Let's start the task. I provide you url link of images "
                    "You evaluate each single PNG image. "
                    "Return only 'id' or 'ood'. Do not include any other text, or explanations."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": "Evaluate this image based on the definition provided."},
                    {"type": "image_url", "image_url": {"url": image_url}}  # ✅ CORRECT

                ]}
            ]
        )

        # Extract response
        assessment = response.choices[0].message.content.strip()

        # Store result
        results.append([image_url, assessment])
        print(f"✅ Processed: {image_url} → {assessment}")

    except Exception as e:
        print(f"❌ Error processing {image_url}: {str(e)}")
        results.append([image_url, "ERROR"])

# Save results to CSV
df_results = pd.DataFrame(results, columns=["Image_URL", "Classification"])
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Results saved to {OUTPUT_CSV}")
