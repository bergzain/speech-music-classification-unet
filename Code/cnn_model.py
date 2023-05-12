#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn as nn
import torch


# In[3]:


# Define Wave-U-Net architecture
class CNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, num_features=40):
        super(CNNModel, self).__init__()

        # Encoding layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
        )

        # Decoding layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
        )

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x



