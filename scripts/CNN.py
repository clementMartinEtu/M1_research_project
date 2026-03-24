import torch.nn as nn

class SmallCNN(nn.Module):
    """
    SmallCNN is a simple convolutional neural network designed for image classification.

    Architecture:
    - 3 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
    - Fully connected classifier with dropout regularization

    Expected input:
        Tensor of shape (batch_size, 3, H, W)
        The spatial size should be such that after 3 MaxPool(2),
        the feature maps are 8x8 (e.g., input 64x64).

    Output:
        Tensor of shape (batch_size, 1)
        Typically used for binary classification (before sigmoid).

    Attributes:
        features (nn.Sequential): Convolutional feature extractor.
        classifier (nn.Sequential): Fully connected classification head.
    """

    def __init__(self):
        """
        Initializes the SmallCNN model by defining its layers.
        """
        super(SmallCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            self.Conv_Block(3, 64),     # Input: 3 channels (RGB)
            self.Conv_Block(64, 128),
            self.Conv_Block(128, 256)
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),  # Assumes input images reduce to 8x8
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),            # Binary classification output
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def Conv_Block(self, in_channels, out_channels):
        """
        Creates a convolutional block consisting of:
        Conv2D -> BatchNorm -> ReLU -> MaxPool

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )