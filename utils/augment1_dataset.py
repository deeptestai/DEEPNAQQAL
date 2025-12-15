import torch
import numpy as np
import os
from PIL import Image

def augment_dataset(
        X, y, augment_threshold, transform,
        model_cls, expected_output, device,
        save_folder=None
    ):
    
    # If the user wants saving, create folder
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_counter = 0

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
            i_sample = y_indexes[np.random.randint(len(y_indexes))]
            sample_to_transform = X[i_sample].to(device)

            aug_image = transform(sample_to_transform)
            aug_image = torch.Tensor(aug_image).to(device)

            # Classifier check
            try:
                if torch.argmax(model_cls(aug_image)).item() == expected_output:
                    continue
                accepted_image = aug_image
            except:
                aug_image = torch.unsqueeze(aug_image, 0)
                if torch.argmax(model_cls(aug_image)).item() == expected_output:
                    continue
                accepted_image = aug_image[0]

            # -----------------------------
            # SAVE IMAGE HERE (new feature)
            # -----------------------------
            if save_folder is not None:
                # Convert [C,H,W] tensor to numpy image
                img_np = accepted_image.detach().cpu().numpy()
                img_np = img_np.transpose(1, 2, 0)  # CHW → HWC

                # If grayscale, remove the channel dim
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)

                # Convert to uint8
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

                # Save to PNG
                img = Image.fromarray(img_np)
                img.save(os.path.join(save_folder, f"aug_{save_counter}.png"))
                save_counter += 1
            # -----------------------------

            # Add to dataset
            X_new.append(accepted_image)
            y_new.append(u_values[i])
            u_counts[i] += 1
            
    X_new = torch.stack(X_new)
    y_new = torch.stack(y_new)

    return torch.vstack([X.cpu(), X_new.cpu()]), torch.vstack([y.cpu(), y_new.cpu()])
