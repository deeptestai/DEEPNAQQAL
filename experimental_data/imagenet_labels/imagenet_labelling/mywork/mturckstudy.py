import csv
import glob
import ntpath
import random
from math import floor

# Paths and URLs
survey_folder = r"/home/maryam/Documents/human-feedback-validity-checker-dnn/label/imagenet_labelling/Mturk-study-pizza/out_img"  # Your folder containing the images
base_url = r"https://tig-imagent-bucket.s3.eu-central-1.amazonaws.com/"
csv_file = r"imagenet_mturk_links.csv"
extra_csv_file = r"imagenet_mturk_links_extra.csv"
acq_url = base_url + "vr_0_e281_pNone.png"

# Get all images and shuffle them randomly
images = glob.glob(survey_folder + '/*')
random.shuffle(images)

# Parameters
number_questions = 10  # Number of images per row
number_surveys = floor(len(images) / number_questions) - 1  # Number of rows
remainder = len(images) % number_questions

# If there's an exact division, adjust the number of surveys and avoid extra file
extra_survey = remainder != 0
if remainder == 0:
    number_surveys = len(images) // number_questions
partitions = [images[i:i + number_questions] for i in range(0, len(images), number_questions)]

# Writing the main CSV file
with open(csv_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Adjust the header to include incremental numbering for acq_question
    header = ["image_url" + str(i + 1) for i in range(number_questions + 1)]
    writer.writerow(header)

    for i in range(number_surveys):
        row = []
        for element in partitions[i]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)  # Add acq_url as the last image URL
        random.shuffle(row)  # Shuffle the row before saving
        writer.writerow(row)

# Writing the extra CSV file if there are leftover images
if extra_survey:
    with open(extra_csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # Adjust the header for the extra file
        header = ["image_url" + str(i + 1) for i in range(len(partitions[-1]) + 1)]
        writer.writerow(header)
        row = []
        for element in partitions[-1]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)  # Add acq_url as the last image URL
        random.shuffle(row)
        writer.writerow(row)

print("CSV files created successfully.")