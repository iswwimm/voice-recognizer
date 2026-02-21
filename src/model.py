import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    """
    CNN-based feature extractor for generating speaker embeddings from Mel-spectrograms.
    """
    def __init__(self):
        super(SpeakerEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

if __name__ == "__main__":
    dummy_input = torch.randn(1, 1, 64, 65)
    
    model = SpeakerEncoder()
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (1, 128), f"Expected output shape (1, 128), got {output.shape}"
    print("Model architecture verified successfully.")