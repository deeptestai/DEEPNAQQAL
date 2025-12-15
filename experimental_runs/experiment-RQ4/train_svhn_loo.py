### Imports
import warnings
import numpy as np
import pandas as pd
import torch
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from path_config import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
#from models.ResNet50_transfer import ResNet50_transfer
from models.vgg16_transfer_svhn import vgg16_transfer_svhn
from models.classifiers.pt_svhn_classifier import SVHN_classifier
from utils.train1 import train_model
from utils.myDataset import myDataset
from utils.args import get_parser
import copy
import os

warnings.filterwarnings("ignore")

### Helper Functions

def create_dataset_dict(csv_file, question_marks="remove"):
    df = pd.read_csv(csv_file)
    if question_marks.lower() == "remove":
        df = df[df["ID/OOD Human"] != "?"]
    elif question_marks.lower() == "replace":
        df["ID/OOD Human"] = df["ID/OOD Human"].replace("?","ood")
    else:
        raise Exception(f"Invalid question_marks value: {question_marks}")
        
    my_df = {"x": [],
             "y": [],
             "y_replaced": [],
             "y_daiv": [],
             "y_so": [],
             "y_deepsvdd": [],
             "y_llm": [],
             "y_llm2": [],
             "y_llm3": [],
             "tool": [],
             "aug": []}
    for i in range(len(df)):
        image_info = df.iloc[i]

        tool = image_info["TOOL"]
        img_id = image_info["ID"]

        images_folder = f"generated_images/{DATASET_NAME}_inputs/{DATASET_NAME}_{tool}"            
        image_loc = f"{images_folder}/{img_id}.npy"
        
        image = np.load(image_loc)[0].transpose((2,0,1))
        my_df["x"].append(image)
        my_df["tool"].append(image_info["TOOL"])
        my_df["y"].append(image_info["ID/OOD Human"])
        my_df["y_daiv"].append(image_info["ID/OOD DAIV"])
        my_df["y_so"].append(image_info["ID/OOD Selforacle 9999"])
        my_df["y_deepsvdd"].append(image_info["ID/OOD DeepSVDD"]) # Added new line
        my_df["y_llm"].append(image_info["ID/OOD LLM"])
        my_df["y_llm2"].append(image_info["ID/OOD LLM2"])
        my_df["y_llm3"].append(image_info["ID/OOD LLM3"])        
    return my_df

### Training Setup
parser = get_parser()
args = parser.parse_args()
BATCH_SIZE = args.bs
RANDOM_SEED = args.seed
K_SPLITS = args.k_splits
SHUFFLE = True
UPSAMPLE_TO = args.class_img
IMG_SIZE = 32
DATASET_NAME = "svhn"
dest_folder = "results/results_ool2_svhn"
SAVE_MODEL = args.save_model
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
QM_POLICY = args.qm
expected_output = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data Preprocessing
# Data Preprocessing
csv_file = f"experimental_data/TIGvalidity - {DATASET_NAME.upper()}_SURVEY.csv"
data = create_dataset_dict(csv_file, question_marks="remove")

X = torch.Tensor(data["x"]).to(device)
# Keep labels as raw strings for encoding inside the loop
y = np.array(data["y"], dtype=str) 
y_so =np.array(data["y_so"], dtype=str) 
y_daiv = np.array(data["y_daiv"], dtype=str) 
y_deepsvdd = np.array(data["y_deepsvdd"], dtype=str)
y_llm = np.array(data["y_llm"], dtype=str)
y_llm2 = np.array(data["y_llm2"], dtype=str) 
y_llm3 = np.array(data["y_llm3"], dtype=str) 
tool = np.array(data["tool"], dtype=str)  # Keep tool names for encoding later


def load_svhn_classifier(model_path, device):
    """
    Load the SVHN classifier model from a saved state dict.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        model_cls (torch.nn.Module): Loaded classifier model ready for inference.
    """
    model_cls = SVHN_classifier()  # Initialize model structure
    model_sd = torch.load(model_path, map_location=device)  # Load saved weights
    model_cls.load_state_dict(model_sd)  # Load state into the model
    model_cls.to(device)  # Move model to the correct device
    model_cls.eval()  # Set model to evaluation mode
    return model_cls

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './models/classifiers/svhn_class.pt'
model_cls = load_svhn_classifier(model_path, device)
def augment_dataset(X, y_tool, augment_threshold, transform, model_cls, expected_output, device):
    if augment_threshold == -1:
        return X.cpu(), y_tool.cpu()

    # Extract labels from y_tool (first column)
    y_labels = y_tool[:, 0]  

    u_values, u_counts = torch.unique(y_labels, return_counts=True)
    X_new, y_new = [], []

    for i in range(len(u_values)):
        y_indexes = (y_labels == u_values[i]).nonzero(as_tuple=True)[0].tolist()
        
        while u_counts[i] < augment_threshold:
             i_sample = y_indexes[np.random.randint(len(y_indexes))]
             aug_image = transform(X[i_sample].to(device)).cpu()

             if torch.argmax(model_cls(aug_image.unsqueeze(0).to(device))).item() == expected_output:
                continue

             X_new.append(aug_image)
             y_new.append(y_tool[i_sample].cpu())  # Keep both label and tool info

             u_counts[i] += 1

    X_new = torch.stack(X_new)
    y_new = torch.stack(y_new)

    return torch.vstack([X.cpu(), X_new]), torch.vstack([y_tool.cpu(), y_new])





folders = ['sv', 'ox', 'dlf', 'dx', 'dj']
# Encode the tool names
le_y = preprocessing.LabelEncoder()
le_y.fit(np.concatenate([y, y_so, y_daiv, y_deepsvdd]))  
# Convert encoded labels to tensors
y_tensor = torch.tensor(le_y.transform(y), dtype=torch.long).to(device)
y_so_tensor = torch.tensor(le_y.transform(y_so), dtype=torch.long).to(device)
y_daiv_tensor = torch.tensor(le_y.transform(y_daiv), dtype=torch.long).to(device)
y_deepsvdd_tensor = torch.tensor(le_y.transform(y_deepsvdd), dtype=torch.long).to(device)
y_llm_tensor = torch.tensor(le_y.transform(y_llm), dtype=torch.long).to(device)
y_llm2_tensor = torch.tensor(le_y.transform(y_llm2), dtype=torch.long).to(device)
y_llm3_tensor = torch.tensor(le_y.transform(y_llm3), dtype=torch.long).to(device)
# Encode tools (folders)
le_tool = preprocessing.LabelEncoder()
le_tool.fit(tool)

# Convert tool labels to tensor
tool_tensor = torch.tensor(le_tool.transform(tool), dtype=torch.long).to(device)
# Ensure original data is not modified across iterations
tool_original = tool.clone() if isinstance(tool, torch.Tensor) else tool.copy()
X_original = X.clone()
y_so_tensor_original = y_so_tensor.clone()
y_daiv_tensor_original = y_daiv_tensor.clone()
y_deepsvdd_tensor_original = y_deepsvdd_tensor.clone()
y_llm_tensor_original = y_llm_tensor.clone()
y_llm2_tensor_original = y_llm2_tensor.clone()
y_llm3_tensor_original = y_llm3_tensor.clone()
y_tensor_original = y_tensor.clone()

for leave_one_out_folder in folders:
    print(f"\n{'='*30}\nLeave-One-Out Testing: {leave_one_out_folder}\n{'='*30}\n")
    # Restore original copies (to avoid data modification across iterations)
    # Ensure original data is stored outside the loop
    # Fresh copy of the original data for each iteration
    # Reset fresh copies for each iteration to prevent modification propagation
    X = X_original.clone()
    y_tensor = y_tensor_original.clone()
    y_so_tensor = y_so_tensor_original.clone()
    y_daiv_tensor = y_daiv_tensor_original.clone()
    y_deepsvdd_tensor = y_deepsvdd_tensor_original.clone()
    y_llm_tensor = y_llm_tensor_original.clone()
    y_llm2_tensor = y_llm2_tensor_original.clone()
    y_llm3_tensor = y_llm3_tensor_original.clone()
    tool = tool_original.copy()
    # Reset NumPy array safely
    # Identify test and train data based on the current leave-out folder
    test_indices = (tool == leave_one_out_folder)
    train_indices = (tool != leave_one_out_folder)
    # Ensure masks have the same length as X
    test_indices = torch.tensor(test_indices, dtype=torch.bool)
    train_indices = torch.tensor(train_indices, dtype=torch.bool)
    # Debugging - Ensure correct split
    print(f"Total samples: {len(tool)}, Test count: {test_indices.sum().item()}, Train count: {train_indices.sum().item()}")

    # Verify mask shapes to prevent indexing errors
   # assert test_indices.shape[0] == X.shape[0], f"Mask shape {test_indices.shape[0]} doesn't match X {X.shape[0]}"
    print("X shape:", X.shape)
    print("Tool shape:", tool.shape)
     # Verify that test set contains only the selected folder
    print(f"Test set tools: {tool[test_indices].tolist()}")
    assert all(t == leave_one_out_folder for t in tool[test_indices].tolist()), f"Error: Test set should contain only '{leave_one_out_folder}'"
    # Verify that train set does NOT contain the test folder
    train_tools = tool[train_indices].tolist()
    print(f"Train set unique tools: {set(train_tools)}")
    assert leave_one_out_folder not in train_tools, f"Error: '{leave_one_out_folder}' found in training set!"

    # Split the data
    X_test, X_train = X[test_indices], X[train_indices]
    y_test, y_train = y_tensor[test_indices], y_tensor[train_indices]
    y_test_so = y_so_tensor[test_indices]
    y_test_daiv = y_daiv_tensor[test_indices]
    y_test_deepsvdd = y_deepsvdd_tensor[test_indices]
    y_test_llm = y_llm_tensor[test_indices]
    y_test_llm2 = y_llm2_tensor[test_indices]
    y_test_llm3 = y_llm3_tensor[test_indices]

    tool_train = tool[train_indices]
    # Print sizes for verification
    print(f"Test Data Size for {leave_one_out_folder}: {X_test.shape[0]}")
    print(f"Train Data Size for {leave_one_out_folder}: {X_train.shape[0]}")
    # Encode train tools to numeric values
    le_tool.fit(tool_train)  # Refit only with training tools
    tool_train_encoded = torch.tensor(le_tool.transform(tool_train), dtype=torch.long).to(device)
    y_tool_train = torch.transpose(torch.stack([y_train, tool_train_encoded], dim=0), 0, 1)
    # Ensure that data exists
    if len(X_test) == 0 or len(X_train) == 0:
        print(f"⚠️ No data found for {leave_one_out_folder}. Skipping...")
        continue

    print(f"Train Data Size for {leave_one_out_folder}: {X_train.shape[0]}")
    print(f"Test Data Size for {leave_one_out_folder}: {X_test.shape[0]}")

    # Augmentation Setup
    transform_augment = T.Compose([
        T.ToPILImage(),
        T.RandomInvert(0.5),
        T.RandomAdjustSharpness(2, p=0.99),
        T.RandomRotation(degrees=20),
        T.RandomResizedCrop(size=IMG_SIZE, scale=(0.75, 0.9)),
        T.ColorJitter(brightness=0.3, hue=0.5),
        T.ToTensor(),
    ])
    #mean = [0.14354469, 0.14354469, 0.14354469]
    #std = [0.29302433, 0.29302433, 0.29302433]
    
    X, y = augment_dataset(X_train, y_tool_train, UPSAMPLE_TO, transform_augment, model_cls, expected_output, device)
    print(f"Unique labels after augmentation: {torch.unique(y)}")
    print(f"Shape of augmented data: {X.shape}")
    print(f"Max index allowed: {len(X) - 1}")
    y=y[:, 0]

    # Training and Validation
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=SHUFFLE)
    crossval_acc = []
    models = []

    # Ensure y_train is on CPU and a NumPy array
    for i, (train_idx, val_idx) in enumerate(skf.split(X.cpu().numpy(), y.cpu().numpy())):
        print(f"{'='*10} Split {i+1}/{K_SPLITS} {'='*10}")

        model_ft, data_transforms = vgg16_transfer_svhn(num_classes=2)
        model_ft = model_ft.to(device)

        train_set = myDataset(X=X[train_idx], y=y[train_idx].cpu().numpy(), transform=data_transforms["train"])

        val_set = myDataset(X=X[val_idx], y=y[val_idx].cpu().numpy(), transform=data_transforms["val"])
        # Define dataset sizes
        dataset_sizes = {
           "train": len(train_set),
           "val": len(val_set)
        }
        dataloaders = {
            "train": torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE),
            "val": torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE),
        }

        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(train_set.get_class_weights()).to(device))
        optimizer_ft = optim.Adam(model_ft.parameters(), weight_decay=0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

        model_ft, best_acc = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=16,dataset_sizes = dataset_sizes, dataloaders=dataloaders, device=device
        )
        crossval_acc.append(best_acc)
        models.append(copy.deepcopy(model_ft.state_dict()))

    # Evaluate on Test Set
    best_model_s = models[np.argmax([acc.item() for acc in crossval_acc])]
    if SAVE_MODEL:
        model_dest_folder = f"validator_models/{DATASET_NAME}/{QM_POLICY}/"
        os.makedirs(model_dest_folder, exist_ok=True)
        model_filename = f"VALIDATOR_{leave_one_out_folder}_SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.pth"
        torch.save(best_model_s, os.path.join(model_dest_folder, model_filename))
    print(f"Best model for folder {leave_one_out_folder} saved as {model_filename}")

    best_model, _ = vgg16_transfer_svhn(num_classes=2)
    best_model.load_state_dict(best_model_s)
    best_model = best_model.to(device)
    best_model.eval()

    y_true = []
    y_pred = []
    # Ensure `y_test` is on the CPU before passing to `myDataset`
    test_set = myDataset(X=X_test.cpu(), y=y_test.cpu().numpy(), transform=data_transforms["val"])

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,  shuffle=SHUFFLE)

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(labels)):
            y_true.append(labels[i].cpu().numpy())
            y_pred.append(preds[i].cpu().numpy())
   # cm = confusion_matrix(y_true, y_pred)
   # acc = accuracy_score(y_true, y_pred)

    # Evaluation Summary Writing
    with open(f"{dest_folder}/SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}_{leave_one_out_folder}.txt", "w") as f:
         f.write(("="*20) + f"{'CLASSIFICATION REPORT':^25}"  + ("="*20))
         f.write("\n")
         f.write(f"{classification_report(y_true, y_pred)}\n")

         f.write(("="*20) + f"{'CONFUSION MATRIX':^25}" + ("="*20))
         f.write("\n")
         cm = confusion_matrix(y_true, y_pred)
         acc = accuracy_score(y_true, y_pred)

         cm_daiv = confusion_matrix(y_test.cpu().numpy(), y_test_daiv.cpu().numpy())
         acc_daiv = accuracy_score(y_test.cpu().numpy(), y_test_daiv.cpu().numpy())

         cm_so = confusion_matrix(y_test.cpu().numpy(), y_test_so.cpu().numpy())
         acc_so = accuracy_score(y_test.cpu().numpy(), y_test_so.cpu().numpy())


         cm_deepsvdd = confusion_matrix(y_test.cpu().numpy(), y_test_deepsvdd.cpu().numpy())  # Add DeepSVDD confusion matrix
         acc_deepsvdd = accuracy_score(y_test.cpu().numpy(), y_test_deepsvdd.cpu().numpy())  # Add DeepSVDD accuracy
         cm_llm = confusion_matrix(y_test.cpu().numpy(), y_test_llm.cpu().numpy())  # Add DeepSVDD confusion matrix
         acc_llm = accuracy_score(y_test.cpu().numpy(), y_test_llm.cpu().numpy())
         cm_llm2 = confusion_matrix(y_test.cpu().numpy(), y_test_llm2.cpu().numpy())  # Add DeepSVDD confusion matrix
         acc_llm2 = accuracy_score(y_test.cpu().numpy(), y_test_llm2.cpu().numpy())
         cm_llm3 = confusion_matrix(y_test.cpu().numpy(), y_test_llm3.cpu().numpy())  # Add llm3 confusion matrix
         acc_llm3 = accuracy_score(y_test.cpu().numpy(), y_test_llm3.cpu().numpy())

         # Write DeepSVDD accuracy and confusion matrix
         #f.write(f"Confusion Matrix for DeepSVDD:\n{cm_deepsvdd}\n")
         #f.write("\n")
         tn, fp, fn, tp = cm.ravel()
         tn_daiv, fp_daiv, fn_daiv, tp_daiv = cm_daiv.ravel()
         tn_so, fp_so, fn_so, tp_so = cm_so.ravel()
         tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd = cm_deepsvdd.ravel()
         tn_llm, fp_llm, fn_llm, tp_llm = cm_llm.ravel()
         tn_llm2, fp_llm2, fn_llm2, tp_llm2 = cm_llm2.ravel()
         tn_llm3, fp_llm3, fn_llm3, tp_llm3 = cm_llm3.ravel()
         # Write confusion matrix and accuracy comparison
         # Write confusion matrix and accuracy comparison
         f.write(f"{'':5}   {'Ours':^10}|{'DAIV':^10}|{'SO':^10}|{'DeepSVDD':^10}|{'LLM':^10}|{'LLM2':^10}|{'LLM3':^10}")
         f.write("\n")
         f.write(f"{'TP':5} = {tp:^10}|{tp_daiv:^10}|{tp_so:^10}|{tp_deepsvdd:^10}|{tp_llm:^10}|{tp_llm2:^10}|{tp_llm3:^10}")
         f.write("\n")
         f.write(f"{'TN':5} = {tn:^10}|{tn_daiv:^10}|{tn_so:^10}|{tn_deepsvdd:^10}|{tn_llm:^10}|{tn_llm2:^10}|{tn_llm3:^10}")
         f.write("\n")
         f.write(f"{'FP':5} = {fp:^10}|{fp_daiv:^10}|{fp_so:^10}|{fp_deepsvdd:^10}|{fp_llm:^10}|{fp_llm2:^10}|{fp_llm3:^10}")
         f.write("\n")
         f.write(f"{'FN':5} = {fn:^10}|{fn_daiv:^10}|{fn_so:^10}|{fn_deepsvdd:^10}|{fn_llm:^10}|{fn_llm2:^10}|{fn_llm3:^10}")
         f.write("\n")
         f.write(f"{'Acc':5} = {acc:^10.3}|{acc_daiv:^10.3}|{acc_so:^10.3}|{acc_deepsvdd:^10.3}|{acc_llm:^10.3}|{acc_llm2:^10.3}|{acc_llm3:^10.3}")
         f.write("\n")

