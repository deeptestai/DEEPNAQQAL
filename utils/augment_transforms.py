import torchvision.transforms as T

def get_augment_transforms(dataset):
    img_size_dict = {'mnist': 28, 'svhn':32, 'imagenet':224}
    IMG_SIZE = img_size_dict.get(dataset)

    if dataset == 'svhn':
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

    elif dataset == 'mnist':
        transform_augment = T.Compose(
                                    [
                                        
                                        T.ToPILImage(),
                                        T.RandomAdjustSharpness(2, p=0.99),
                                        T.Lambda(lambda x: T.RandomRotation(degrees=20)(x)), \
                                                                            #fill=T.ToTensor()(x).mean(dim=(1,2)).tolist()  )(x) ),
                                        T.RandomResizedCrop(size=IMG_SIZE, scale=(0.75, 0.9)),
                                        
                                        T.ToTensor()
                                    ]
                                )
    elif dataset == 'imagenet':
        transform_augment = T.Compose(
                                    [
                                        
                                        T.ToPILImage(),
                                        
                                        T.RandomAdjustSharpness(2, p=0.99),
                                        T.Lambda(lambda x: T.RandomRotation(degrees=20)(x)), \
                                                                            #fill=T.ToTensor()(x).mean(dim=(1,2)).tolist()  )(x) ),
                                        T.RandomResizedCrop(size=IMG_SIZE, scale=(0.75, 0.9)),
                                        T.ColorJitter(brightness=0.1, hue=0.1),
                                        T.ToTensor()

                                        
                                    ]
                                )
    else:
        raise Exception(f"Argument 'dataset' value (={dataset}) unrecognised")

    return transform_augment
