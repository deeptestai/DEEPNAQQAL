#  How to Run LLM-Validator

This section provides instructions for running the LLM-based validators—**GPT** and **LLaVA**—to classify images as **valid** or **invalid**.


###  1. Running GPT Validators

The **GPT** model is used to validate images by processing **URLs** of the images. To run the validator for a specific dataset (MNIST, SVHN, or ImageNet), execute:

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

> Note:Both GPT and LLaVA validators use the same prompt logic, tailored to the dataset selected.The key difference is in input type: GPT expects URLs pointing to images.LLaVA works with local .png image files.This dual setup allows consistent validation across models using different input modalities.

### 3. LLM Validator Prompts

### System Prompt (MNIST)

Imagine you are an expert image assessor. Provide an evaluation of validity based on this definition: ``valid'' means recognizable by humans as part of the input domain, e.g., if an image of handwritten digits belongs to the domain of handwritten digits. If the image I provide you actually represents an image from the domain of handwritten digits, you answer `id'. If the image does not represent a handwritten digit, you answer `ood'. Return only `id' or `ood'. Do not include any other text, or explanations.

---

### User Prompt

Evaluate this image based on the provided definition of validity.




