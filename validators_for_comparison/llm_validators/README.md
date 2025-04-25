#  How to Run LLM-Validator

This section provides instructions to run the LLM-based validators—**GPT-4.0** and **LLaVA**—for classifying images as **valid** or **invalid**.


###  1. Running GPT-4.0 Validator

The **GPT-4.0** model is used to validate images by processing **URLs** of the images. To run the validator for a specific dataset (MNIST, SVHN, or ImageNet), execute:

```
python3 <dataset-name>_gpt.py
```
Replace <dataset-name> with one of the following: mnist, svhn, or imagenet.This script will classify each image as valid or invalid using GPT-4.0's image understanding capability via URL input.

###  2. Running LLaVA Validator

The LLaVA model requires image inputs in .png format rather than URLs. To run the LLaVA validator:

 a. Convert all .npy files inside the generated_image folder into .png format.

 b. Then, run the validation script for your chosen dataset:
 
```
 python3 <dataset-name>_llava.py
```
Replace <dataset-name> with one of the following: mnist, svhn, or imagenet.

> Note:Both GPT-4.0 and LLaVA validators use the same prompt logic, tailored to the dataset selected.The key difference is in input type: GPT-4.0 expects URLs pointing to images.LLaVA works with local .png image files.This dual setup allows consistent validation across models using different input modalities.






