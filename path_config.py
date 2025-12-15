import os
import sys

# Absolute path to the project root directory
ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path so imports work from any subfolder
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)   # insert at front to prioritize root

# Add important folders to Python path so imports always work
for folder in ["models", "utils", "validators_for_comparison"]:
    full_path = os.path.join(ROOT, folder)
    if full_path not in sys.path:
        sys.path.append(full_path)
os.chdir(ROOT)
# --- Define all important subpaths in ONE place ---
# Absolute path to the project root directory

# --- Define all important subpaths in ONE place ---

# Data
EXPERIMENTAL_DATA = os.path.join(ROOT, "experimental_data")
IMAGENET_LABELS = os.path.join(EXPERIMENTAL_DATA, "imagenet_labels")
IMAGENET_CSV = os.path.join(IMAGENET_LABELS, "imagenet_labelling", "data2.csv")
IMAGENET_IMAGES = os.path.join(IMAGENET_LABELS, "imagenet_labelling", "images")
# Experimental runs
EXPERIMENT_RQ1_2 = os.path.join(ROOT, "experimental_runs", "experiment_RQ1-2")
EXPERIMENT_RQ3 = os.path.join(ROOT, "experimental_runs", "experiment_RQ3")
EXPERIMENT_RQ4 = os.path.join(ROOT, "experimental_runs", "experiment_RQ4")
EXPERIMENT_RQ5 = os.path.join(ROOT, "experimental_runs", "experiment_RQ5")

# Validators
VALIDATORS = os.path.join(ROOT, "validators_for_comparison")
DEEPSVDD = os.path.join(VALIDATORS, "DeepSvdd_validators")
LLM_VALIDATORS = os.path.join(VALIDATORS, "llm_validators")
RECONST_VALIDATORS = os.path.join(VALIDATORS, "reconst_based_validators")

# Generated images
GENERATED_IMAGES = os.path.join(ROOT, "generated_images")

# Models
MODELS = os.path.join(ROOT, "models")

# Utils
UTILS = os.path.join(ROOT, "utils")

# Results
RESULTS = os.path.join(ROOT, "results")
LLM_LABEL_RESULTS = os.path.join(RESULTS, "llm-label-results")
COMBINED_MAP_CODE = os.path.join(RESULTS, "combined_map_code")
RESULTS_MWU = os.path.join(RESULTS, "results_mwu")
# Helper to get dataset CSV absolute path
# Automatically set working directory to ROOT
# Patch os.path.join globally so relative image paths resolve automatically
# This ensures scripts using relative paths from CSVs (like label/imagenet_labelling/images/xxx.npy)
# always point to the correct absolute location
_original_join = os.path.join
def _patched_join(*args):
    # If first arg is relative and looks like it starts with 'label/imagenet_labelling/images'
    if len(args) >= 2 and args[0] == "label" and args[1] == "imagenet_labelling":
        return _original_join(IMAGENET_LABELS, *args[2:])
    return _original_join(*args)

os.path.join = _patched_join
os.chdir(ROOT)
