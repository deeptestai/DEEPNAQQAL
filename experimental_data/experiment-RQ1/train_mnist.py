### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import copy

# Models for validator
from models.lenet1_transfer import lenet1_transfer
from models.svhn_transfer import svhn_transfer
from models.vgg16_transfer_mnist import vgg16_transfer_mnist
# My utils
from utils.train import train_model
from utils.myDataset import myDataset
from utils.augment_transforms import get_augment_transforms
from utils.get_classifier import get_classifier
from utils.create_dataset_dict import create_dataset_dict
from utils.create_imagenet_dataset import create_imagenet_dataset
from utils.augment_dataset import augment_dataset
from utils.args import get_parser

import warnings
import os
warnings.filterwarnings("ignore")



### PARSER 
parser = get_parser()
args = parser.parse_args()

get_model_to_train = lenet1_transfer   # you can shift model vgg16_transfer_mnist as well

UPSAMPLE_TO = args.class_img #minimum samples per class
BATCH_SIZE = args.bs
RANDOM_SEED = args.seed
K_SPLITS = args.k_splits
QM_POLICY = args.qm
SHUFFLE = True
SAVE_MODEL = args.save_model

DATASET_NAME = args.dataset
dest_folder = f"results/{DATASET_NAME}/results_{QM_POLICY}/test_mnist"

expected_output = {"svhn": 5,
                   "mnist": 5,
                   "imagenet": 963} # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("Num per class ", UPSAMPLE_TO)
print("Seed ", RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#################################### GET DATA ####################################
data = create_dataset_dict(f"experimental_data/TIGvalidity - {DATASET_NAME.upper()}_SURVEY.csv", 
                           dataset_name=DATASET_NAME,
                           question_marks=QM_POLICY)
print(f"Number of samples in dataset: {len(data['x'])}")
X = torch.Tensor(data["x"])
y = data["y"]
y_so = data["y_so"]
y_daiv = data["y_daiv"]
y_deepsvdd = data["y_deepsvdd"]  # DeepSVDD predictions (added new column)
y_llm =data["y_llm"]
y_llm2 =data["y_llm2"]
tool = data["tool"]

################################ GET ENCODED LABELS ################################
le_y = preprocessing.LabelEncoder()
le_y.fit(y)
print(le_y.classes_)
print(le_y.transform(["id"]))

y = torch.Tensor( le_y.transform(y) ).to(torch.long)
y_so = torch.Tensor( le_y.transform(y_so) ).to(torch.long)
y_daiv = torch.Tensor( le_y.transform(y_daiv) ).to(torch.long)
y_deepsvdd = torch.Tensor(le_y.transform(y_deepsvdd)).to(torch.long)  # Encode DeepSVDD predictions
y_llm = torch.Tensor(le_y.transform(y_llm)).to(torch.long)
y_llm2 = torch.Tensor(le_y.transform(y_llm2)).to(torch.long)
le_tool = preprocessing.LabelEncoder()
le_tool.fit(tool)
tool = torch.Tensor( le_tool.transform(tool) ).to(torch.long)

################################ GET TEST INDEXES ################################
### I need both of them to do an equal split over id/ood and over the TIGs
if DATASET_NAME == "mnist":
    y_tool = torch.transpose(torch.stack([y, tool], dim=0), 0, 1)
    indexes = list(range(len(y_tool)))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_tool shape: {y_tool.shape}")

    _, index_test, _, _ = train_test_split(indexes, y_tool,\
                                        test_size=0.3, stratify= y_tool,\
                                        shuffle=SHUFFLE, random_state=RANDOM_SEED)

    X_test = X[index_test]
    y_test = y[index_test]
    y_test_so = y_so[index_test]
    y_test_daiv = y_daiv[index_test]
    y_test_deepsvdd = y_deepsvdd[index_test]  # Added: DeepSVDD test labels
    y_test_llm = y_llm[index_test]
    y_test_llm2 = y_llm2[index_test]
    mask = np.ones(len(X), bool)
    mask[index_test] = False
    X_train = X[mask]
    y_train = y[mask]
    y_tool_train = y_tool[mask]
    #y_train_deepsvdd = y_deepsvdd[mask]  # Added: DeepSVDD training labels
else:
   raise ValueError(f"Dataset '{DATASET_NAME}' is not supported.")


########################## AUGMENT AND EQUALIZE TRAIN SET ########################## 

# Load the classifier model
model_cls = get_classifier(DATASET_NAME)
model_cls.to(device)

transform_augment = get_augment_transforms(dataset=DATASET_NAME)
cls_model = get_classifier(dataset=DATASET_NAME)
cls_model = cls_model.to(device)

# import matplotlib.pyplot as plt
# # plt.imshow()
# # plt.show()
# tr_img = np.random.randint(len(X_train))
# tr_img = np.random.randint(len(X_train))
# print(torch.max(X_train[ tr_img ]))
# print(torch.min(X_train[ tr_img ]))
# input()
# # plt.imshow(X_test[np.random.randint(len(X_test))])
# # plt.show()
# tr_img = np.random.randint(len(X_test))
# print(torch.max(X_train[ tr_img ]))
# print(torch.min(X_train[ tr_img ]))
# input()

X, y = augment_dataset(X_train, y_tool_train, UPSAMPLE_TO, transform_augment, 
                       cls_model, expected_output=expected_output[DATASET_NAME], device=device)
y = y[:, 0]
print("Trainset shape:", X.shape)
print("Trainset y:")
print(pd.Series(y).value_counts())

### Aiut per debugging
# import matplotlib.pyplot as plt
# randi = np.random.randint(0,60)
# plt.imshow(X[randi].T)
# plt.show()
# input()

# img = X_test[randi]
# print(img.shape)
# # img = img[[2,1,0], :, :]
# plt.imshow(img.T)
# plt.show()
# input()


########################## START CROSSVAL ########################## 
crossval_acc = []
models = []
skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=SHUFFLE)
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
    print("dataset_sizes:",dataset_sizes)
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

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

    model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                    num_epochs=1, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
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
    model_dest_folder = f"validator_models/{DATASET_NAME}/{QM_POLICY}/lenet_mnist_testlllm"
    isExist = os.path.exists(model_dest_folder)
    if not isExist:
        os.makedirs(model_dest_folder)    
    torch.save(best_model_sd, f"{model_dest_folder}/VALIDATOR_SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.pth")

best_model, _ = get_model_to_train(num_classes=2)
best_model.load_state_dict(best_model_sd)
best_model = best_model.to(device)

y_true = []
y_pred = []



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
        
# Add comparison for DeepSVDD
#cm_deepsvdd = confusion_matrix(y_true_deepsvdd, y_pred)  # Confusion matrix for DeepSVDD
#acc_deepsvdd = accuracy_score(y_true_deepsvdd, y_pred)  # Accuracy for DeepSVDD

isExist = os.path.exists(dest_folder)
if not isExist:
   os.makedirs(dest_folder)

with open(f"{dest_folder}/SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.txt", "w") as f:
    # Write classification report
    f.write(("="*20) + f"{'CLASSIFICATION REPORT':^25}" + ("="*20))
    f.write("\n")
    f.write(classification_report(y_true, y_pred))

    # Write DeepSVDD accuracy and confusion matrix
    #f.write(f"DeepSVDD Accuracy = {acc_deepsvdd:.3f}\n")
    #f.write(f"Confusion Matrix for DeepSVDD:\n{cm_deepsvdd}\n")
    #f.write("\n")

    # Compute confusion matrices and accuracies for each method
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

    tn, fp, fn, tp = cm.ravel()
    tn_daiv, fp_daiv, fn_daiv, tp_daiv = cm_daiv.ravel()
    tn_so, fp_so, fn_so, tp_so = cm_so.ravel()

    # Compute DeepSVDD confusion matrix components
    tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd =cm_deepsvdd.ravel()
    tn_llm, fp_llm, fn_llm, tp_llm = cm_llm.ravel()
    tn_llm2, fp_llm2, fn_llm2, tp_llm2 = cm_llm2.ravel()
    # Write table of results
    f.write(("="*20) + f"{'CONFUSION MATRIX':^25}" + ("="*20))
    f.write("\n")
    f.write(f"{'':6}   {'Ours':^10}|{'DAIV':^10}|{'SO':^10}|{'DeepSVDD':^10}|{'LLM':^10}|{'LLM2':^10}")
    f.write("\n")
    f.write(f"{'TP':6} = {tp:^10}|{tp_daiv:^10}|{tp_so:^10}|{tp_deepsvdd:^10}|{tp_llm:^10}|{tp_llm2:^10}")
    f.write("\n")
    f.write(f"{'TN':6} = {tn:^10}|{tn_daiv:^10}|{tn_so:^10}|{tn_deepsvdd:^10}|{tn_llm:^10}|{tn_llm2:^10}")  # Fixed here
    f.write("\n")
    f.write(f"{'FP':6} = {fp:^10}|{fp_daiv:^10}|{fp_so:^10}|{fp_deepsvdd:^10}|{fp_llm:^10}|{fp_llm2:^10}")
    f.write("\n")
    f.write(f"{'FN':6} = {fn:^10}|{fn_daiv:^10}|{fn_so:^10}|{fn_deepsvdd:^10}|{fn_llm:^10}|{fn_llm2:^10}")
    f.write("\n")
    f.write(f"{'Acc':6} = {acc:^10.3f}|{acc_daiv:^10.3f}|{acc_so:^10.3f}|{acc_deepsvdd:^10.3f}|{acc_llm:^10.3f}|{acc_llm2:^10.3f}")  # Fixed floating point precision format
    f.write("\n")

