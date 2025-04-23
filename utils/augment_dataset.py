import torch
import numpy as np

def augment_dataset(X, y, augment_threshold, transform, model_cls, expected_output, device):
    if augment_threshold == -1:
        return X.cpu(), y.cpu()

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

            aug_image = transform(sample_to_transform)
            aug_image = torch.Tensor( aug_image ).to(device)
            try:
                if torch.argmax(model_cls(aug_image)).item() == expected_output: continue
                X_new.append(aug_image)
            except: 
                aug_image = torch.unsqueeze(aug_image, 0)
                if torch.argmax(model_cls(aug_image)).item() == expected_output: continue
                X_new.append(aug_image[0])
            y_new.append(u_values[i])
            u_counts[i] += 1
            
    X_new = torch.stack(X_new)
    y_new = torch.stack(y_new)
    
    return torch.vstack([X.cpu(), X_new.cpu()]), torch.vstack([y.cpu(), y_new.cpu()])
