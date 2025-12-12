### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from path_config import *

import torchvision
import torchvision.transforms as T

from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
from utils.train import train_model
import copy

# Models for validator
#from models.ResNet50_transfer import ResNet50_transfer
from models.lenet1_transfer import lenet1_transfer
from models.svhn_transfer import svhn_transfer
from models.vgg16_transfer_svhn import vgg16_transfer_svhn

# Models for classifiers
from models.classifiers.pt_svhn_classifier import SVHN_classifier

from utils.myDataset import myDataset

import warnings
import os
from utils.args import get_parser
warnings.filterwarnings("ignore")


# TODO: Move to utils !!!
def create_dataset_dict(csv_file, question_marks="remove"):
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
        
    my_df = {"x": [],
             "y": [],
             "y_replaced": [],
             "y_daiv": [],
             "y_so": [],
             "y_deepsvdd": [],   #added new lin
             "y_llm":[],
             "y_llm2":[],
             "y_llm3":[],
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
        my_df["y_so"].append(image_info["ID/OOD Selforacle 99"])
        my_df["y_deepsvdd"].append(image_info["ID/OOD DeepSVDD"]) # Added new line
        my_df["y_llm"].append(image_info["ID/OOD LLM"])
        my_df["y_llm2"].append(image_info["ID/OOD LLM2"])
        my_df["y_llm3"].append(image_info["ID/OOD LLM3"])
    return my_df

def augment_dataset(X, y, augment_threshold, transform):
    if augment_threshold == -1:
        return X.cpu(), y.cpu()
    # Load the classifier model
    filename = './models/classifiers/svhn_class.pt'
    model = SVHN_classifier()
    model_sd = torch.load(filename)
    model.load_state_dict(model_sd)
    model.to(device)

    u_values, u_counts = torch.unique(y, return_counts=True, dim=0)
    num_u_values = len(u_values)
    X_new = []
    y_new = []
    
    for i in range(num_u_values):
        # get all indexes for that class
        y_indexes = [k for k in range(len(y)) if all(y[k] == u_values[i])]
        
        while u_counts[i] < augment_threshold:
            i_sample = y_indexes[np.random.randint(len(y_indexes), )]
            sample_to_transform = X[i_sample]
            sample_to_transform = sample_to_transform.to(device)

            aug_image = transform(sample_to_transform).to(device)
            if torch.argmax(model(aug_image)).item() == 5: continue
                
            X_new.append(aug_image)
            y_new.append(u_values[i])
            u_counts[i] += 1
            
    X_new = torch.stack(X_new)
    y_new = torch.stack(y_new)
    
    return torch.vstack([X.cpu(), X_new.cpu()]), torch.vstack([y.cpu(), y_new.cpu()])


parser = get_parser()
args = parser.parse_args()

get_model_to_train = svhn_transfer

IMG_SIZE = 32
DATASET_NAME = "svhn"
dest_folder = "results/svhn/results_remove/llm3_svhn"

augment_up_to = args.class_img #minimum samples per class
BATCH_SIZE = args.bs
RANDOM_SEED = args.seed
K_SPLITS = args.k_splits
SHUFFLE = True
QM_POLICY = args.qm
SAVE_MODEL = args.save_model
UPSAMPLE_TO = args.class_img #minimum samples per class

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("Num per class ", augment_up_to)
print("Seed ", RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform_augment = T.Compose(
    [
        
        T.ToPILImage(),
        T.RandomInvert(0.5),
        T.RandomAdjustSharpness(2, p=0.99),
        T.Lambda(lambda x: T.RandomRotation(degrees=20)(x)), \
                                            #fill=T.ToTensor()(x).mean(dim=(1,2)).tolist()  )(x) ),
        T.RandomResizedCrop(size=IMG_SIZE, scale=(0.75, 0.9)),
        T.ColorJitter(brightness=0.3, hue=0.5),
                    #    contrast=0.5), \
                        # saturation=0.5),
        T.ToTensor()
    ]
)

mean = [0.14354469, 0.14354469, 0.14354469]
std = [0.29302433, 0.29302433, 0.29302433]

#################################### GET DATA ####################################
data = create_dataset_dict(csv_file = f"experimental_data/TIGvalidity - {DATASET_NAME.upper()}_SURVEY.csv", question_marks=QM_POLICY)

X = torch.Tensor(data["x"])
y = data["y"]
y_so = data["y_so"]
y_daiv = data["y_daiv"]
y_deepsvdd = data["y_deepsvdd"]
y_llm = data["y_llm"]
y_llm2 = data["y_llm2"]
y_llm3 = data["y_llm3"]
tool = data["tool"]

################################ GET ENCODED LABELS ################################
le_y = preprocessing.LabelEncoder()
le_y.fit(y)

y = torch.Tensor(
            le_y.transform(y)    
        ).to(torch.long)

y_so = torch.Tensor(
                le_y.transform(y_so)
            ).to(torch.long)

y_daiv = torch.Tensor(
                le_y.transform(y_daiv)
            ).to(torch.long)
y_deepsvdd = torch.Tensor(le_y.transform(y_deepsvdd)).to(torch.long)
y_llm = torch.Tensor(le_y.transform(y_llm)).to(torch.long)
y_llm2 = torch.Tensor(le_y.transform(y_llm2)).to(torch.long)
y_llm3 = torch.Tensor(le_y.transform(y_llm3)).to(torch.long)
le_tool = preprocessing.LabelEncoder()
le_tool.fit(tool)
tool = torch.Tensor(
        le_tool.transform(tool)
    ).to(torch.long)

################################ GET TEST INDEXES ################################
### I need both of them to do an equal split over id/ood and over the TIGs
y_tool = torch.transpose(torch.stack([y, tool], dim=0), 0, 1)
indexes = list(range(len(y_tool)))
_, index_test, _, _ = train_test_split(indexes, y_tool,\
                                       test_size=0.3, stratify= y_tool,\
                                       shuffle=SHUFFLE, random_state=RANDOM_SEED)

X_test = X[index_test]
y_test = y[index_test]
y_test_so = y_so[index_test]
y_test_daiv = y_daiv[index_test]
y_test_deepsvdd = y_deepsvdd[index_test]  # Split DeepSVDD labels
y_test_llm = y_llm[index_test]
y_test_llm2 = y_llm2[index_test]
y_test_llm3 = y_llm3[index_test]  
mask = np.ones(len(X), bool)
mask[index_test] = False
X_train = X[mask, : , : , :]
y_train = y[mask]
y_tool_train = y_tool[mask]

########################## AUGMENT AND EQUALIZE TRAIN SET ########################## 
print(X_train.shape)
X, y = augment_dataset(X_train, y_tool_train, augment_up_to, transform_augment)
y = y[:, 0]
print(X.shape)


########################## START CROSSVAL ########################## 
crossval_acc = []
models = []
skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=False)
for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"{'='*10} Split {i+1}/{K_SPLITS} {'='*10}")
    ################ MODEL ################
    model_ft, data_transforms = get_model_to_train(num_classes=2)
    model_ft = model_ft.to(device)
    
    ################# DATA #################
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_index]
    y_val = y[val_index]
    
    train_set = myDataset(X=X_train, y=y_train, transform=data_transforms["train"])
    val_set = myDataset(X=X_val, y=y_val, transform=data_transforms["val"])
    
    dataset_sizes = {
        "train": len(train_set),
        "val": len(val_set)
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE),
        "val": torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    }

    
    ################# CRTIERION & OPTIMIZER #################
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(train_set.get_class_weights()))
    criterion =    criterion.to(device)

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), weight_decay=0)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

    model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                    num_epochs=16, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                                    device=device)
    
    crossval_acc.append(best_acc)
    models.append(copy.deepcopy(model_ft.state_dict()))



########################## TEST BEST MODEL ########################## 
accuracies = [x.item() for x in crossval_acc]
print(accuracies)
print(sum(accuracies)/len(accuracies))

index_best = np.argmax(accuracies)
best_model_sd = models[index_best]
if SAVE_MODEL:
    model_dest_folder = f"/home/vincenzo.riccio/human-feedback-validity-checker-dnn/validator_models/{DATASET_NAME}/{QM_POLICY}/test_llm3"
    isExist = os.path.exists(model_dest_folder)
    if not isExist:
        os.makedirs(model_dest_folder)    
    torch.save(best_model_sd, f"{model_dest_folder}/VALIDATOR_SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.pth")
best_model, _ = get_model_to_train(num_classes=2)
best_model.load_state_dict(best_model_sd)
best_model = best_model.to(device)

y_true = []
y_pred = []
#y_pred_deepsvdd = []  # Add storage for DeepSVDD predictions

test_set = myDataset(X=X_test, y=y_test, transform=data_transforms["val"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=SHUFFLE)
best_model.eval()

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = best_model(inputs)
                    
    _, preds = torch.max(outputs, 1)
    
    for i in range(len(labels)):
        y_true.append(labels[i].cpu().numpy())
        y_pred.append(preds[i].cpu().numpy())
        #y_pred_deepsvdd.append(y_test_deepsvdd[i].cpu().numpy())  # Add DeepSVDD predictions
isExist = os.path.exists(dest_folder)
if not isExist:
   os.makedirs(dest_folder)

with open(f"{dest_folder}/SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.txt", "w") as f:
    f.write(("="*20) + f"{'CLASSIFICATION REPORT':^25}"  + ("="*20))
    f.write("\n")
    f.write(classification_report(y_true, y_pred))

    f.write(("="*20) + f"{'CONFUSION MATRIX':^25}" + ("="*20))
    f.write("\n")
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    cm_daiv = confusion_matrix(y_test, y_test_daiv)
    acc_daiv = accuracy_score(y_test, y_test_daiv)

    cm_so = confusion_matrix(y_test, y_test_so)
    acc_so = accuracy_score(y_test, y_test_so)

    cm_deepsvdd = confusion_matrix(y_test, y_test_deepsvdd)  # Add DeepSVDD confusion matrix
    acc_deepsvdd = accuracy_score(y_test, y_test_deepsvdd)  # Add DeepSVDD accuracy
    cm_llm = confusion_matrix(y_test, y_test_llm)  # Add DeepSVDD confusion matrix
    acc_llm = accuracy_score(y_test, y_test_llm)
    cm_llm2 = confusion_matrix(y_test, y_test_llm2)  # Add DeepSVDD confusion matrix
    acc_llm2 = accuracy_score(y_test, y_test_llm2)
    cm_llm3 = confusion_matrix(y_test, y_test_llm3)  # Add DeepSVDD confusion matrix
    acc_llm3 = accuracy_score(y_test, y_test_llm3)


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

