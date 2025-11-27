import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from collections import Counter
import copy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.train1 import train_model
from utils.myDataset import myDataset
from utils.args import get_parser
#from utils.create_dataset_dict_img import create_dataset_dict
from utils.augment_transforms import get_augment_transforms
from utils.get_classifier import get_classifier
from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
#from utils.train import train_model
import copy

# Models for validator
from models.ResNet50_transfer import ResNet50_transfer
from models.lenet1_transfer import lenet1_transfer
from models.svhn_transfer import svhn_transfer
#from models.vgg16_transfer import vgg16_transfer
#from models.vgg16_transfer_mnist import vgg16_transfer_mnist
from models.ResNet152_transfer import ResNet152_transfer
# My utils
from utils.myDataset import myDataset
from utils.augment_transforms import get_augment_transforms
from utils.get_classifier import get_classifier
from utils.create_dataset_dict_img import create_dataset_dict
#from utils.create_imagenet_dataset import create_imagenet_dataset
from utils.augment_dataset import augment_dataset
from models.vgg16_transfer import vgg16_transfer
# Models for validator
#from models.vgg16_transfer import vgg16_transfer_svhn
import warnings
warnings.filterwarnings("ignore")

# Argument parser
parser = get_parser()
args = parser.parse_args()
get_model_to_train = vgg16_transfer
UPSAMPLE_TO = args.class_img #minimum samples per class
BATCH_SIZE = args.bs
RANDOM_SEED = args.seed
K_SPLITS = args.k_splits
QM_POLICY = args.qm
SHUFFLE = True
SAVE_MODEL = args.save_model
DATASET_NAME = args.dataset
dest_folder = f"results/{DATASET_NAME}/results_{QM_POLICY}/test_ool2_70"
os.makedirs(dest_folder, exist_ok=True)
expected_output = {"svhn": 5,
                   "mnist": 5,
                   "imagenet": 963} # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("Num per class ", UPSAMPLE_TO)
print("Seed ", RANDOM_SEED)

df = pd.read_csv("./experimental_data/imagenet_labels/imagenet_labelling/data2.csv")
subfolders = ['dx', 'dlf', 'ox', 'sv']
print(df.head())  # Display first few rows
def process_folder(folder_name):
    print(f"\n{'='*30}\nProcessing: {folder_name}\n{'='*30}\n")
    
    data_df = df[df["TOOL"] == folder_name]
    
    data = create_dataset_dict(data_df, dataset_name=DATASET_NAME, question_marks=QM_POLICY)
    #test_data = create_dataset_dict(test_df, dataset_name=DATASET_NAME, question_marks=QM_POLICY)
    
    X = torch.Tensor(data["x"])
    y = data["y"]  # Replace `y` with processed values
    y_so = data["y_so"]
    y_daiv = data["y_daiv"]
    y_deepsvdd = data["y_deepsvdd"]
    y_llm = data["y_llm"]
    y_llm2 = data["y_llm2"]
    y_llm3 = data["y_llm3"]
    tool = data["tool"]

   # X_test = torch.Tensor(test_data["x"]).to(device)
   # y_test = torch.Tensor(test_data["y"]).to(torch.long)
   # y_so = torch.Tensor(test_data["y_so"]).to(torch.long)
   # y_daiv = torch.Tensor(test_data["y_daiv"]).to(torch.long)
   # y_deepsvdd = torch.Tensor(test_data["y_deepsvdd"]).to(torch.long)
   # y_llm = torch.Tensor(test_data["y_llm"]).to(torch.long)
   # y_llm2 = torch.Tensor(test_data["y_llm2"]).to(torch.long)
     
    # Label Encoding
    # Combine all labels before fitting the LabelEncoder
    all_labels = np.concatenate([y, y_daiv, y_so, y_deepsvdd, y_llm, y_llm2, y_llm3])

    le_y = preprocessing.LabelEncoder()
    le_y.fit(all_labels)
    print(le_y.classes_)
    print(le_y.transform(["id"]))

    y = torch.Tensor( le_y.transform(y) ).to(torch.long)
    y_so = torch.Tensor( le_y.transform(y_so) ).to(torch.long)
    y_daiv = torch.Tensor( le_y.transform(y_daiv) ).to(torch.long)
    y_deepsvdd = torch.Tensor(le_y.transform(y_deepsvdd)).to(torch.long)
    y_llm = torch.Tensor(le_y.transform(y_llm)).to(torch.long)
    y_llm2 = torch.Tensor(le_y.transform(y_llm2)).to(torch.long) 
    y_llm3 = torch.Tensor(le_y.transform(y_llm3)).to(torch.long) 
    le_tool = preprocessing.LabelEncoder()
    le_tool.fit(tool)
    tool = torch.Tensor( le_tool.transform(tool) ).to(torch.long)
    print("Label encoding and test set selection completed.")
    y_tool = torch.transpose(torch.stack([y, tool], dim=0), 0, 1)
    # **Keep Your Existing Indexing**
    indexes = list(range(len(y_tool)))
    # Identify OOD instances
    ood_label = le_y.transform(["ood"])[0]
    ood_indices = (y.numpy() == ood_label).nonzero()[0]
    id_indices = (y.numpy() != ood_label).nonzero()[0]

    # Print ID and OOD sample count
    num_id = len(id_indices)
    num_ood = len(ood_indices)
    print(f"Total ID samples: {num_id}")
    print(f"Total OOD samples: {num_ood}")

    # Exit training if no OOD samples are found
    if num_ood == 0:
        print("No OOD instances found. Exiting training.")
        return None 

    # If only one OOD instance exists, force it into training
    if num_ood == 1:
       print("Only one OOD instance found. Keeping it in the training set.")

       # Perform normal stratified split **only for ID samples**
       index_id = list(set(range(len(y))) - set(ood_indices))  # Exclude OOD from stratification
       index_train, index_test, _, _ = train_test_split(
          index_id, y.numpy()[index_id], test_size=0.3, stratify=y.numpy()[index_id], random_state=RANDOM_SEED
       )

       # Ensure the **only** OOD sample stays in training
       index_train = np.concatenate([index_train, ood_indices])

       print(f" Train Set Size: {len(index_train)}, Test Set Size: {len(index_test)}")

    # If we have both ID and OOD balanced, perform stratified splitting
    else:
        print("OOD and ID are balanced. Performing stratified train-test split.")
        index_train, index_test, _, _ = train_test_split(
            list(range(len(y))), y, test_size=0.3, stratify=y, shuffle=True, random_state=RANDOM_SEED
        )
    # 🔹 **Check If Stratification is Possible**
    #unique_combinations, counts = np.unique(y_tool, axis=0, return_counts=True)
   # if len(unique_combinations) == 1:  
    #   print("Skipping folder: Only one unique class with a single instance.")
     #  return  # Skip the folder
   # if any(counts < 2):  # If any class has only one instance
    #    print("Warning: Some classes have only one instance. Disabling stratify for this folder.")
     #   stratify_option = None
   # else:
    #    stratify_option = y_tool

    # 🔹 **Perform Train-Test Split (KEEPING YOUR INDEXES)**
   # if stratify_option is not None:
    #    _, index_test, _, _ = train_test_split(indexes, y_tool, test_size=0.3, stratify=stratify_option, shuffle=SHUFFLE, random_state=RANDOM_SEED)
   # else:
      #  _, index_test, _, _ = train_test_split(indexes, y_tool, test_size=0.3, stratify=y_tool, shuffle=SHUFFLE, random_state=RANDOM_SEED)
     # Extract train and test datasets
    

    X_test = X[index_test]
    y_test = y[index_test]
    y_test_so = y_so[index_test]
    y_test_daiv = y_daiv[index_test]
    y_test_deepsvdd = y_deepsvdd[index_test]  # Split DeepSVDD labels
    y_test_llm = y_llm[index_test]
    y_test_llm2 = y_llm2[index_test]
    y_test_llm3 = y_llm3[index_test]
    #mask = np.ones(len(X), bool)
    #mask[index_test] = False
    X_train = X[index_train]
    y_train = y[index_train]
    y_tool_train = y_tool[index_train]
    # Check OOD count before augmentation
    num_ood_before_aug = (y_train.numpy() == ood_label).sum()
    print(f" OOD Samples Before Augmentation: {num_ood_before_aug}")
    # Data augmentation
    transform_augment = get_augment_transforms(dataset=DATASET_NAME)
    cls_model = get_classifier(dataset=DATASET_NAME).to(device)
    X_train, y_train = augment_dataset(X_train,  y_tool_train, UPSAMPLE_TO, transform_augment, cls_model, expected_output= expected_output[DATASET_NAME], device=device)
    y_train = y_train[:, 0]
    print("Trainset shape:", X_train.shape)
    print("Trainset y:")
    print(pd.Series(y_train.cpu().numpy()).value_counts())
    # Check how many OOD samples exist after augmentation
    num_ood_after_aug = (y_train.numpy() == ood_label).sum()
    print(f" OOD Samples After Augmentation: {num_ood_after_aug}")
    # 🔺 Print the difference to verify augmentation worked
    print(f" Augmented OOD Samples: {num_ood_after_aug - num_ood_before_aug}")

    # K-Fold Training
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_SEED)
    crossval_acc = []
    models = []
    
    for i, (train_index, val_index) in enumerate(skf.split(X_train.cpu(), y_train.cpu())):
        print(f"{'='*10} Split {i+1}/{K_SPLITS} {'='*10}")
        model_ft, data_transforms = get_model_to_train(num_classes=len(torch.unique(y_train)))
        model_ft = model_ft.to(device)

        train_set = myDataset(X=X_train[train_index], y=y_train[train_index], transform=data_transforms["train"])
        val_set = myDataset(X=X_train[val_index], y=y_train[val_index], transform=data_transforms["val"])
        
        dataloaders = {
            "train": torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE),
            "val": torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        }
        
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), weight_decay=0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)
        
        model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=16, dataloaders=dataloaders, dataset_sizes={"train": len(train_set), "val": len(val_set)}, device=device)
        
        crossval_acc.append(best_acc)
        models.append(copy.deepcopy(model_ft.state_dict()))
    
    
    accuracies = [x.item() for x in crossval_acc]
    print(accuracies)
    print(sum(accuracies)/len(accuracies))

    index_best = np.argmax(accuracies)
    best_model_sd = models[index_best]

    if SAVE_MODEL:
       model_dest_folder = f"/home/vincenzo.riccio/human-feedback-validity-checker-dnn/validator_models/{DATASET_NAME}/{QM_POLICY}/test_ool70"
       isExist = os.path.exists(model_dest_folder)
       if not isExist:
           os.makedirs(model_dest_folder)    
       torch.save(best_model_sd, f"{model_dest_folder}/VALIDATOR_SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.pth")

    best_model, _ = get_model_to_train(num_classes=2)

    best_model.load_state_dict(best_model_sd)
    best_model = best_model.to(device)
    
    y_true, y_pred = [], []
    test_set = myDataset(X=X_test, y=y_test, transform=data_transforms["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=SHUFFLE)
    best_model.eval()
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    with open(f"{dest_folder}/SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}_{folder_name}.txt", "w") as f:
        f.write(("="*20) + f"{'CLASSIFICATION REPORT':^25}"  + ("="*20))
        f.write("\n")
        f.write(classification_report(y_true, y_pred))

        f.write(("="*20) + f"{'CONFUSION MATRIX':^25}" + ("="*20))
        f.write("\n")
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        acc = accuracy_score(y_true, y_pred)

        cm_daiv = confusion_matrix(y_test, y_test_daiv, labels=[0,1])
        acc_daiv = accuracy_score(y_test, y_test_daiv)

        cm_so = confusion_matrix(y_test, y_test_so,labels=[0,1])
        acc_so = accuracy_score(y_test, y_test_so)

        cm_deepsvdd = confusion_matrix(y_test, y_test_deepsvdd,labels=[0,1])  # Add DeepSVDD confusion matrix
        acc_deepsvdd = accuracy_score(y_test, y_test_deepsvdd)  # Add DeepSVDD accuracy
        cm_llm = confusion_matrix(y_test, y_test_llm,labels=[0,1])  # Add LLM confusion matrix
        acc_llm = accuracy_score(y_test, y_test_llm)
        cm_llm2 = confusion_matrix(y_test, y_test_llm2,labels=[0,1])  # Add LLM2 confusion matrix
        acc_llm2 = accuracy_score(y_test, y_test_llm2)
        cm_llm3 = confusion_matrix(y_test, y_test_llm3,labels=[0,1])  # Add LLM3 confusion matrix
        acc_llm3 = accuracy_score(y_test, y_test_llm3)
        # Initialize metrics with default values
        tn, fp, fn, tp = 0, 0, 0, 0
        tn_daiv, fp_daiv, fn_daiv, tp_daiv = 0, 0, 0, 0
        tn_so, fp_so, fn_so, tp_so = 0, 0, 0, 0
        tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd = 0, 0, 0, 0
        tn_llm, fp_llm, fn_llm, tp_llm = 0 ,0 ,0 ,0 
        tn_llm2, fp_llm2, fn_llm2, tp_llm2 = 0, 0, 0, 0
        tn_llm3, fp_llm3, fn_llm3, tp_llm3 = 0, 0, 0, 0


        # Handle confusion matrix shapes
        if cm.shape == (2, 2):
           tn, fp, fn, tp = cm.ravel()

        if cm_daiv.shape == (2, 2):
           tn_daiv, fp_daiv, fn_daiv, tp_daiv = cm_daiv.ravel()

        if cm_so.shape == (2, 2):
           tn_so, fp_so, fn_so, tp_so = cm_so.ravel()

        if cm_deepsvdd.shape == (2, 2):
            tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd = cm_deepsvdd.ravel()
        if cm_llm.shape == (2, 2):
            tn_llm, fp_llm, fn_llm, tp_llm = cm_llm.ravel()
        if cm_llm2.shape == (2, 2):
            tn_llm2, fp_llm2, fn_llm2, tp_llm2 = cm_llm2.ravel()
        if cm_llm3.shape == (2, 2):
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
    
for folder in subfolders:
    process_folder(folder)
