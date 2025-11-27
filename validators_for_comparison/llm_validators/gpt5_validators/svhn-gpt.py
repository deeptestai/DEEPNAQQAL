import openai
import pandas as pd
import os
import time

# Print API key from environment
print("Environment Key:", os.getenv("OPENAI_API_KEY"))

# Set API key explicitly for OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to your CSV file (Modify this path)
CSV_FILE_PATH = "svhn_link.csv"
df = pd.read_csv(CSV_FILE_PATH, header=0)
df = df.applymap(lambda x: str(x)
                 .strip()
                 .replace("\n","")
                 .replace("\r","")
                 .replace(" ","")
                 .replace(">","")
                 .replace("<",""))

print(df.head(10))


# Flatten to a simple list
image_urls = df.values.flatten()

# Remove null/empty cells
image_urls = [u for u in image_urls if u.startswith("http")]

# Load the CSV file
#df = pd.read_csv(CSV_FILE_PATH, header=None, names=["Image_URL"])

# Store results
results = []


# Iterate through each image URL
for image_url in image_urls:                    #df["Image_URL"]:
    attempt = 0
    while attempt < 5:  # Retry up to 4 times for errors
        try:
            response = client.chat.completions.create(
                model="gpt-5.1",    # change model name gpt-40, it will work for gpt version 40 rest of code will remain the same
                temperature=0.0,
                messages=[
                    {"role": "system", "content": (
                        "Imagine you are an expert image assessor. "
                        "You have to provide an evaluation of validity based on this definition: "
                        "'Valid means recognizable by humans as part of the input domain, "
                        "e.g., if an image of house number belongs to the domain of house number'.\n\n"
                        "I provide you urls of images. If the image actually represents an image from the domain of house number,you answer 'id'."
                        " If the image does not represent a house number, you answer 'ood'.\n\n"
                        "Return only 'id' or 'ood'. Do not include any other text, or explanations."
                    )},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Evaluate this image based on the definition provided."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ]
            )

            # Extract response
            #assessment = clean_response(response.choices[0].message.content)
            # Extract response
            assessment = response.choices[0].message.content.strip()

            # Store result
            results.append([image_url, assessment])
            print(f" Processed: {image_url} → {assessment}")
            break  # Exit retry loop if successful

        except Exception as e:
            print(f" Error processing {image_url} (Attempt {attempt+1}): {str(e)}")
            attempt += 1
            time.sleep(2)  # Wait before retrying

            if attempt == 3:
                results.append([image_url, "ERROR"])

# Save results to CSV
OUTPUT_CSV = "svhn_classified_gpt5.1_run2.csv"
df_results = pd.DataFrame(results, columns=["Image_URL", "Classification"])
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"\n Results saved to {OUTPUT_CSV}")
