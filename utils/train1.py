import time
import torch
import os
import sys
from tempfile import TemporaryDirectory  # Add this import
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, patience=5):
    since = time.time()

    model.to(device)
    best_acc = 0.0
    epochs_no_improve = 0  # Counter to track epochs without improvement

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    # Display loading bar
                    if phase == "train":
                        sys.stdout.write('\r')
                        j = (i + 1) / len(dataloaders[phase])
                        sys.stdout.write("[%-20s] %s%%" % ('=' * int(20 * j),  f"{(100 * j):.4}"))
                        sys.stdout.flush()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == "train": 
                    print()
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Early stopping logic
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                        epochs_no_improve = 0  # Reset counter if improvement
                    else:
                        epochs_no_improve += 1

                    # Stop training if no improvement for 'patience' epochs
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch} epochs.")
                        model.load_state_dict(torch.load(best_model_params_path))
                        time_elapsed = time.time() - since
                        print(f'Training stopped early in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best val Acc: {best_acc:.4f}')
                        return model, best_acc

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, best_acc
