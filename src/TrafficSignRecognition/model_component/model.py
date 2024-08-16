import torch
import torch.nn as nn

class TrafficSignRecognitionModel(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shpae: int):

        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = int(hidden_units * 2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = int(hidden_units * 2), out_channels = int(hidden_units * 2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = int(hidden_units * 2), out_channels = int(hidden_units * 4), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = int(hidden_units * 4), out_channels = int(hidden_units * 4), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = int(hidden_units * 4), out_channels = int(hidden_units * 8), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = int(hidden_units * 8) * 4 * 4, out_features = 1024),
            nn.Linear(in_features = 1024, out_features = 128),
            nn.Linear(in_features = 128, out_features = 196),
            nn.Linear(in_features = 196, out_features = output_shpae)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
