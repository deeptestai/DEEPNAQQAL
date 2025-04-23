import pandas as pd 
import numpy as np

def create_dataset_dict(input_data, dataset_name, question_marks="remove"):
    if isinstance(input_data, str):  # If a CSV file path is passed
        print(f"Reading CSV file: {input_data}")
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):  # If a DataFrame is passed
        print("Processing provided DataFrame...")
        df = input_data
    else:
       raise ValueError("Invalid input: Expected a CSV file path or a DataFrame")

    if question_marks.lower() == "remove":
        df = df[df["ID/OOD Human"] != "?"]
        print("Question marks removed")
        print("End shape: ", df.shape)
    elif question_marks.lower() == "replace":
        df["ID/OOD Human"] = df["ID/OOD Human"].replace("?","ood")
        print("Question marks replaced")
        print("End shape: ", df.shape)
    else:
        raise Exception(f"Question marks should be managed either by 'remove' or 'replace', not {question_marks}")

    df_labels = df["TOOL"] + "_" + df["ID/OOD Human"]
    label_count = df_labels.value_counts() 
    label_count = label_count[label_count <= 1]
    for tool_label in label_count.index:
        tool, label = tool_label.split("_")
        mask = (df["TOOL"] == tool) & (df["ID/OOD Human"] == label)
        df = df[~mask]
        print(f"Rows after removing question marks: {df.shape[0]}")
        print(f"Labels with only 1 sample: {len(label_count)}")
        print(f"Rows after removing labels with 1 sample: {df.shape[0]}")


    df_labels = df["TOOL"] + "_" + df["ID/OOD Human"]


    my_df = {"x": [],
             "y": [],
             "y_replaced": [],
             "y_daiv": [],
             "y_so": [],
             "y_deepsvdd": [],   #added new line
             "y_llm":[],
             "y_llm2":[],
             "tool": [],
             "aug": []}
    for i in range(len(df)):
        image_info = df.iloc[i]

        tool = image_info["TOOL"]
        img_id = image_info["ID"]

        images_folder = f"generated_images/{dataset_name}_inputs/{dataset_name}_{tool}"            
        image_loc = f"{images_folder}/{img_id}.npy"
        
        image = np.load(image_loc)[0].transpose((2,0,1))
        if dataset_name == "imagenet": 
            mean = np.array([0.485, 0.456,0.406]) 
            mean = mean.reshape(-1,1,1)

            image = image[::-1,:,:]  # BGR to RGB, 
            image = (image / 255) + mean  # put it in (0,1) range

        my_df["x"].append(image)
        my_df["tool"].append(image_info["TOOL"])
        my_df["y"].append(image_info["ID/OOD Human"])
        my_df["y_daiv"].append(image_info["ID/OOD DAIV"])
        my_df["y_so"].append(image_info["ID/OOD Selforacle 9999"])
        my_df["y_deepsvdd"].append(image_info["ID/OOD DeepSVDD"]) # Added new line
        my_df["y_llm"].append(image_info["ID/OOD LLM"])
        my_df["y_llm2"].append(image_info["ID/OOD LLM2"])

    return my_df


