import pandas as pd 
import numpy as np

def create_imagenet_dataset(csv_file, question_marks):
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
    label_count = label_count[label_count <=1] # switch <=1 if ood and id both present otherwise <1
    for tool_label in label_count.index:
        tool, label = tool_label.split("_")
        mask = (df["TOOL"] == tool) & (df["ID/OOD Human"] == label)
        df = df[~mask]
    
    df_labels = df["TOOL"] + "_" + df["ID/OOD Human"]


    my_df = {"x": [],
             "y": [],
             "tool": []}
    
    for i in range(len(df)):
        image_info = df.iloc[i]

        tool = image_info["TOOL"]
        file = image_info["file"]

        images_folder = f"label/imagenet_labelling/images"            
        image_loc = f"{images_folder}/{file}"
        
        image = np.load(image_loc)[0].transpose((2,0,1)) # channels first for pyTorch
        image = image[:,:,::-1] # BGR to RGB

        my_df["x"].append(image)
        my_df["tool"].append(image_info["TOOL"])
        my_df["y"].append(image_info["ID/OOD Human"])
        
    return my_df["x"], my_df["y"], my_df["tool"]

