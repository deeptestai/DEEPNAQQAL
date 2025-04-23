import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet50Autoencoder(nn.Module):
    def __init__(self, rep_dim=128):
        super(ResNet50Autoencoder, self).__init__()
        self.rep_dim = rep_dim

        # Encoder - ResNet50 backbone
        base_model = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])  # Use ResNet50 up to its last convolutional block
        self.fc1 = nn.Linear(2048, self.rep_dim)  # Fully connected layer to reduce dimensionality

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, 2048)  # Fully connected layer to expand back to ResNet output size
        self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)  # Output 3 channels (RGB)

        # Initialize decoder weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for the decoder layers."""
        for layer in [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')

    def encode(self, x):
        """Encode input image to latent representation."""
        x = self.encoder(x)  # ResNet50 feature extractor
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling to reduce spatial dimensions to (1, 1)
        x = torch.flatten(x, 1)  # Flatten the tensor
        z = self.fc1(x)  # Map to latent space
        return z

    def decode(self, z):
        # Decoder
        x = self.fc2(z)
        x = x.view(x.size(0), 2048, 1, 1)  # Reshape for deconvolution
        x = F.interpolate(x, scale_factor=7, mode='bilinear', align_corners=False)  # Match ResNet output spatial dimensions (7x7)
        x = F.leaky_relu(self.bn1(self.deconv1(x)))
        x = F.leaky_relu(self.bn2(self.deconv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv3(x)))
        x = F.leaky_relu(self.bn4(self.deconv4(x)))
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Force final output to 224x224
        x = torch.sigmoid(self.deconv5(x))  # Ensure output is in range [0, 1]
        return x        

    def forward(self, x, return_latent=False):
        """
        Forward pass.
        - During autoencoder training (`self.training` is True), return the reconstructed image.
        - During Deep SVDD training or evaluation (`self.training` is False), return only the latent representation.
        """
        z = self.encode(x)  # Compute latent representation
        print(f"Latent representation (z) size: {z.size()}")  # Print latent size
        if return_latent:  # Autoencoder training
            return z  # Return reconstructed image
        else:  # SVDD training or inference
            x_reconstructed = self.decode(z)  # Reconstruct the input
            print(f"Reconstructed image (x_reconstructed) size: {x_reconstructed.size()}")  # Print reconstructed image size
            return x_reconstructed  # Return reconstructed image
