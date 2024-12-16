import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_linear import KANLinear

class self_attention(nn.Module):
    def __init__(self, in_channels):
        super(self_attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Queries, Keys, Values
        queries = self.query_conv(x).view(batch_size, -1, H * W)  # (B, C//8, H*W)
        keys = self.key_conv(x).view(batch_size, -1, H * W)       # (B, C//8, H*W)
        values = self.value_conv(x).view(batch_size, -1, H * W)   # (B, C, H*W)

        # Attention map
        attention = torch.bmm(queries.permute(0, 2, 1), keys)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)              # Normalize attention

        # Apply attention
        out = torch.bmm(values, attention.permute(0, 2, 1))   # (B, C, H*W)
        out = out.view(batch_size, C, H, W)                   # Reshape to (B, C, H, W)

        # Weighted sum of original and attended features
        out = self.gamma * out + x
        return out


class cnn_kan_with_attention(nn.Module):
    def __init__(self):
        super(cnn_kan_with_attention, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Attention Block
        self.attention = self_attention(64)

        # Fully Connected Layers with KANLinear
        self.flatten = nn.Flatten()
        self.kan1 = kan_linear(64 * 8 * 8, 256)
        self.kan2 = kan_linear(256, 10)

    def forward(self, x):
        # Convolutional layers
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)

        # Attention block
        x = self.attention(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        return x
