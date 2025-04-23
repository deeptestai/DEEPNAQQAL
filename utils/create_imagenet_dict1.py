import pandas as pd 
import numpy as np

def create_dataset_dict(csv_file, dataset_name, question_marks="remove"):
    df = pd.read_csv(csv_file)
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
    
    df_labels = df["TOOL"] + "_" + df["ID/OOD Human"]


    my_df = {"x": [],
             "y": [],
             "y_replaced": [],
             "y_daiv": [],
             "y_so": [],
             "y_deepsvdd": [],
             "tool": [],
             "aug": []}
    for i in range(len(df)):
        image_info = df.iloc[i]

        tool = image_info["TOOL"]
        img_id = image_info["file"]

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
    return my_df
