#  Automated Validator (DoLa and Self-Oracle)

Both **DoLa** and **Self-Oracle** validators use an **encoder-decoder (VAE-based) training architecture** to evaluate image validity. Below are the commands and structure to run each validator for different datasets (MNIST, SVHN, and ImageNet).


###  Running DoLa Validator

To run the **DoLa validator**, use:

```
python3 validity_check_dola.py
```
Each dataset folder (mnist, svhn, imagenet) includes a dataset-specific version of this script with the same name.

These scripts perform image validity checks using the trained Dola model.

🔗 To download the pre-trained VAE models for MNIST, SVHN, and ImageNet, click here ()

### Running Self-Oracle Validator
To run the Self-Oracle validator, use:
```
python3 validity_check_selforacle.py
```
Each dataset folder also contains a dataset-specific version of this script (with the same name), tailored to perform validity checks for that specific dataset.
