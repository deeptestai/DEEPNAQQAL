import pandas as pd
import numpy as np
import os

def create_imagenet_dataset(data_source, dataset_name,question_marks, leave_one_out_folder=None):
    # Load CSV if data_source is a file path
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        raise TypeError("data_source must be a CSV file path or a DataFrame")

    # Handle question marks
    if question_marks.lower() == "remove":
        df = df[df["ID/OOD Human"] != "?"]
        print("Question marks removed")
    elif question_marks.lower() == "replace":
        df["ID/OOD Human"] = df["ID/OOD Human"].replace("?", "ood")
        print("Question marks replaced")
    else:
        raise Exception(f"Invalid question_marks value: {question_marks}")

    # Filter out the leave-one-out folder if specified
    if leave_one_out_folder:
        df = df[df["TOOL"] != leave_one_out_folder]

    # Prepare the dataset
    my_df = {"x": [], "y": [], "tool": []}

    for _, image_info in df.iterrows():
        tool = image_info["TOOL"]
        file = image_info["file"]

        # Define image path
        images_folder = "label/imagenet_labelling/images"
        image_loc = f"{images_folder}/{file}"
        #image_loc = os.path.join(images_folder, file)

        if os.path.exists(image_loc):
            image = np.load(image_loc)[0].transpose((2, 0, 1))  # Channels first for PyTorch
            #image = image[:, :, ::-1]  # BGR to RGB
            if dataset_name == "imagenet": 
               mean = np.array([0.485, 0.456,0.406]) 
               mean = mean.reshape(-1,1,1)

               image = image[::-1,:,:]  # BGR to RGB, 
               image = (image / 255) + mean  # put it in (0,1) range



            # Store image and labels
            my_df["x"].append(image)
            my_df["tool"].append(tool)
            my_df["y"].append(image_info["ID/OOD Human"])
        else:
            print(f"File not found: {image_loc}. Skipping...")

    return my_df["x"], my_df["y"], my_df["tool"]

