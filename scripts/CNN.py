import torch.nn as nn

class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()

        self.features = nn.Sequential(
            self.Conv_Block(3, 64),
            self.Conv_Block(64, 128),
            self.Conv_Block(128, 256)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def Conv_Block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )