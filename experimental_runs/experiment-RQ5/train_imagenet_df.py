# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import copy
import warnings
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from path_config import *
# Models for validator
#from models.ResNet152_transfer import ResNet152_transfer
#from models.ResNet50_transfer import ResNet50_transfer
from models.lenet1_transfer import lenet1_transfer
from models.svhn_transfer import svhn_transfer
from models.vgg16_transfer import vgg16_transfer
from models.vgg16_transfer_svhn import vgg16_transfer_svhn
from models.vgg16_transfer_mnist import vgg16_transfer_mnist
# My utils
from utils.train import train_model
from utils.myDataset import myDataset
from utils.augment_transforms import get_augment_transforms
from utils.get_classifier import get_classifier
#from utils.create_dataset_dict import create_dataset_dict
from utils.create_imagenet_dict2 import create_dataset_dict
from utils.augment_dataset import augment_dataset
from utils.args import get_parser

warnings.filterwarnings("ignore")

# Parser
parser = get_parser()
args = parser.parse_args()

get_model_to_train = vgg16_transfer

UPSAMPLE_TO = args.class_img  # Minimum samples per class
BATCH_SIZE = args.bs
RANDOM_SEED = args.seed
K_SPLITS = args.k_splits
QM_POLICY = args.qm
SHUFFLE = True
SAVE_MODEL = args.save_model

DATASET_NAME = args.dataset
dest_folder = f"results/{DATASET_NAME}/results_{QM_POLICY}/img_inc"
os.makedirs(dest_folder, exist_ok=True)
expected_output = {
    "svhn": 5,
    "mnist": 5,
    "imagenet": 963  # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
}

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("Num per class ", UPSAMPLE_TO)
print("Seed ", RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Get Test Indexes
if DATASET_NAME == "imagenet":

    data = create_dataset_dict("./experimental_data/imagenet_labels/imagenet_labelling/data2.csv", dataset_name="imagenet",question_marks=QM_POLICY
    )
    print(f"Number of samples in dataset: {len(data['x'])}")
    X = torch.Tensor(data["x"])
    y = data["y"]
    y_so = data["y_so"]
    y_daiv = data["y_daiv"]
    y_deepsvdd = data["y_deepsvdd"]  # DeepSVDD predictions (added new column)
    #y_llm = data["y_llm"]
    tool = data["tool"]

    # Get Encoded Labels
    le_y = preprocessing.LabelEncoder()
    le_y.fit(y)
    print(le_y.classes_)
    print(le_y.transform(["id"]))

    y = torch.Tensor(le_y.transform(y)).to(torch.long)
    y_so = torch.Tensor(le_y.transform(y_so)).to(torch.long)
    y_daiv = torch.Tensor(le_y.transform(y_daiv)).to(torch.long)
    y_deepsvdd = torch.Tensor(le_y.transform(y_deepsvdd)).to(torch.long)  # Encode DeepSVDD predictions
   # y_llm = torch.Tensor(le_y.transform(y_llm)).to(torch.long)
    le_tool = preprocessing.LabelEncoder()
    le_tool.fit(tool)
    tool = torch.Tensor(le_tool.transform(tool)).to(torch.long)

    y_tool = torch.transpose(torch.stack([y, tool], dim=0), 0, 1)
    indexes = list(range(len(y_tool)))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_tool shape: {y_tool.shape}")

    _, index_test, _, _ = train_test_split(
        indexes, y_tool,
        test_size=0.3, stratify=y_tool,
        shuffle=SHUFFLE, random_state=RANDOM_SEED
    )

    X_test = X[index_test]
    y_test = y[index_test]
    y_test_so = y_so[index_test]
    y_test_daiv = y_daiv[index_test]
    y_test_deepsvdd = y_deepsvdd[index_test]  # Added: DeepSVDD test labels
    #y_test_llm =y_llm[index_test]
    mask = np.ones(len(X), bool)
    mask[index_test] = False
    X_train = X[mask]
    y_train = y[mask]
    y_tool_train = y_tool[mask]


# Augment and Equalize Train Set
# Load the classifier model
model_cls = get_classifier(DATASET_NAME)
model_cls.to(device)

transform_augment = get_augment_transforms(dataset=DATASET_NAME)
cls_model = get_classifier(dataset=DATASET_NAME)
cls_model = cls_model.to(device)

X, y = augment_dataset(
    X_train, y_tool_train, UPSAMPLE_TO, transform_augment,
    cls_model, expected_output=expected_output[DATASET_NAME], device=device
)
y = y[:, 0]
print("Trainset shape:", X.shape)
print("Trainset y:")
print(pd.Series(y).value_counts())


# Define the fractions of the dataset to use for training
training_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10%, 20%, 50%, and 100% of the data

# Directory to save the results
results_dir = "training_size_results"
os.makedirs(results_dir, exist_ok=True)

# Ensure reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Iterate over each training fraction
for fraction in training_fractions:
    print(f"\n{'='*15} Training with {int(fraction*100)}% of Data {'='*15}\n")
    
    # Determine the number of samples for the current training fraction
    num_samples = int(len(X) * fraction)
    print("number of samples",num_samples)        
    # Randomly select a subset of indices for the current training fraction
    subset_indices = np.random.choice(len(X), num_samples, replace=False)
    X_subset = X[subset_indices]
    y_subset = y[subset_indices]
    
    # Initialize variables to store cross-validation results
    crossval_acc = []
    models = []
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_SEED)
    
    # Perform Stratified K-Fold Cross-Validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_subset, y_subset)):
        print(f"Fold {fold + 1}/{K_SPLITS}")
        
        # Split the data into training and validation sets
        X_train_fold = X_subset[train_index]
        y_train_fold = y_subset[train_index]
        X_val_fold = X_subset[val_index]
        y_val_fold = y_subset[val_index]
        
        # Initialize the model
        model_ft, data_transforms = get_model_to_train(num_classes=2)
        model_ft = model_ft.to(device)
        
        # Create datasets and dataloaders
        train_set = myDataset(X=X_train_fold, y=y_train_fold, transform=data_transforms["train"])
        val_set = myDataset(X=X_val_fold, y=y_val_fold, transform=data_transforms["val"])
        
        dataloaders = {
            "train": torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE),
            "val": torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        }
        
        dataset_sizes = {
            "train": len(train_set),
            "val": len(val_set)
        }
        if len(torch.unique(torch.tensor(y_train_fold))) == 1:  # Only one class present
           criterion = nn.CrossEntropyLoss(weight=None)  # Ignore weights for this case
        else:
           criterion = nn.CrossEntropyLoss(weight = torch.Tensor(train_set.get_class_weights()))
           criterion =    criterion.to(device)#

        optimizer_ft = optim.Adam(model_ft.parameters(), weight_decay=0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)
        
        # Train the model
        model_ft, best_acc = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler,
            num_epochs=16, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
            device=device
        )
        
        # Store the best accuracy and model state
        crossval_acc.append(best_acc)
        models.append(copy.deepcopy(model_ft.state_dict()))
    
    ########################## TEST BEST MODEL ########################## 
    #accuracies = [x.item() for x in crossval_acc]
    accuracies = [x.item() if isinstance(x, torch.Tensor) else x for x in crossval_acc]
    print(accuracies)
    print(sum(accuracies)/len(accuracies))

    index_best = np.argmax(accuracies)
    best_model_sd = models[index_best]

    if SAVE_MODEL:
       model_dest_folder = f"validator_models/{DATASET_NAME}/{QM_POLICY}/mnist_inc"
       isExist = os.path.exists(model_dest_folder)
       if not isExist:
          os.makedirs(model_dest_folder)    
          torch.save(best_model_sd, f"{model_dest_folder}/VALIDATOR_SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.pth")

    best_model, _ = get_model_to_train(num_classes=2)
    best_model.load_state_dict(best_model_sd)
    best_model = best_model.to(device)

    y_true = []
    y_pred = []

    # Calculate the average accuracy across all folds
    avg_acc = sum(crossval_acc) / len(crossval_acc)
    print(f"Average Accuracy for {int(fraction*100)}% training data: {avg_acc:.4f}")
    
    # Save the results to a file
    results_file = os.path.join(dest_folder, f"results_{int(fraction*100)}_percent.txt")
    with open(results_file, "w") as f:
        f.write(f"Training Fraction: {int(fraction*100)}%\n")
        f.write(f"Average Accuracy: {avg_acc:.4f}\n")
        f.write("Fold Accuracies:\n")
        for fold, acc in enumerate(crossval_acc, 1):
            f.write(f"  Fold {fold}: {acc:.4f}\n")
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
    cm_deepsvdd = confusion_matrix(y_test, y_test_deepsvdd)  # Confusion matrix for DeepSVDD
    acc_deepsvdd = accuracy_score(y_test, y_test_deepsvdd)  # Accuracy for DeepSVDD

    isExist = os.path.exists(dest_folder)
    if not isExist:
       os.makedirs(dest_folder)
    
    with open(f"{dest_folder}/SEED{RANDOM_SEED}_AUG{UPSAMPLE_TO}.txt", "a") as f:
        f.write("\n" + "="*30 + f" Training Percentage: {int(fraction * 100)}% " + "="*30 + "\n")
        # Write classification report
        #f.write(("="*20) + f"{'CLASSIFICATION REPORT':^25}" + ("="*20))
        #f.write("\n")
        #f.write(classification_report(y_true, y_pred))

        # Write DeepSVDD accuracy and confusion matrix
       # f.write(f"DeepSVDD Accuracy = {acc_deepsvdd:.3f}\n")
       # f.write(f"Confusion Matrix for DeepSVDD:\n{cm_deepsvdd}\n")
       # f.write("\n")

        # Compute confusion matrices and accuracies for each method
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        cm_daiv = confusion_matrix(y_test, y_test_daiv)
        acc_daiv = accuracy_score(y_test, y_test_daiv)

        cm_so = confusion_matrix(y_test, y_test_so)
        acc_so = accuracy_score(y_test, y_test_so)
       # cm_llm = confusion_matrix(y_test, y_test_llm)
       # acc_llm = accuracy_score(y_test, y_test_llm)

        # Initialize metrics with default values
        tn, fp, fn, tp = 0, 0, 0, 0
        tn_daiv, fp_daiv, fn_daiv, tp_daiv = 0, 0, 0, 0
        tn_so, fp_so, fn_so, tp_so = 0, 0, 0, 0
        tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd = 0, 0, 0, 0
        #tn_llm, fp_llm, fn_llm, tp_llm = 0, 0, 0, 0
        # Handle confusion matrix shapes
        if cm.shape == (2, 2):
           tn, fp, fn, tp = cm.ravel()

        if cm_daiv.shape == (2, 2):
           tn_daiv, fp_daiv, fn_daiv, tp_daiv = cm_daiv.ravel()

        if cm_so.shape == (2, 2):
           tn_so, fp_so, fn_so, tp_so = cm_so.ravel()

        if cm_deepsvdd.shape == (2, 2):
            tn_deepsvdd, fp_deepsvdd, fn_deepsvdd, tp_deepsvdd = cm_deepsvdd.ravel()

        #if cm_llm.shape == (2, 2):
         #   tn_llm, fp_llm, fn_llm, tp_llm = cm_llm.ravel()


        # Write table of results
        f.write(("="*20) + f"{'CONFUSION MATRIX':^25}" + ("="*20))
        f.write("\n")
        f.write(f"{'':5}   {'Ours':^10}|{'DAIV':^10}|{'SO':^10}|{'DeepSVDD':^10}")
        f.write("\n")
        f.write(f"{'TP':5} = {tp:^10}|{tp_daiv:^10}|{tp_so:^10}|{tp_deepsvdd:^10}")
        f.write("\n")
        f.write(f"{'TN':5} = {tn:^10}|{tn_daiv:^10}|{tn_so:^10}|{tn_deepsvdd:^10}")
        f.write("\n")
        f.write(f"{'FP':5} = {fp:^10}|{fp_daiv:^10}|{fp_so:^10}|{fp_deepsvdd:^10}")
        f.write("\n")
        f.write(f"{'FN':5} = {fn:^10}|{fn_daiv:^10}|{fn_so:^10}|{fn_deepsvdd:^10}")
        f.write("\n")
        f.write(f"{'Acc':5} = {acc:^10.3}|{acc_daiv:^10.3}|{acc_so:^10.3}|{acc_deepsvdd:^10.3}")
        f.write("\n")

