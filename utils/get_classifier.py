import torch
from models.classifiers import pt_svhn_classifier, pt_lenet1, pt_vgg16

def get_classifier(dataset):
    supported_datasets = ["svhn", "mnist", "imagenet"]
    if dataset not in supported_datasets:
        raise Exception(f"Dataset {dataset} not recognised! It has to be one of the following: {supported_datasets}")
    
    
    filename_dict = {"svhn":"/home/vincenzo.riccio/human-feedback-validity-checker-dnn/models/classifiers/svhn_class.pt",
                     "mnist": "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/models/classifiers/lenet1_class.pt",
                     "imagenet": "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/models/classifiers/imagenet_class.pt"}
                     
    model_classes_dict = {"svhn": pt_svhn_classifier.SVHN_classifier,
                         "mnist":   pt_lenet1.LeNet1,
                        "imagenet":   pt_vgg16.ImageNet_classifier}

    model_cls = model_classes_dict[dataset]()
    model_sd = torch.load(filename_dict[dataset])

    model_cls.load_state_dict(model_sd)
    return model_cls
