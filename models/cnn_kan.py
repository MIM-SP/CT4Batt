import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_linear import KANLinear
from cnn_attention import attention2D

class cnn_kan_with_attention(nn.Module):
    def __init__(self):
        super(cnn_kan_with_attention, self).__init__()
        
        # Convolutional Layers with dynamic attention
        self.conv1 = cnn2D(1, 32, kernel_size=3, padding=1, num_kernels=32)  # Match num_kernels to output channels
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = cnn2D(32, 64, kernel_size=3, padding=1, num_kernels=64)  # Match num_kernels to output channels
        self.pool2 = nn.MaxPool2d(2)

        # Self-Attention Block
        self.attention = attention2D(in_channels=64, ratio=0.25, num_kernels=64, temperature=34)

        # Fully Connected Layers with KANLinear
        self.flatten = nn.Flatten()
        self.kan1 = KANLinear(
            in_features=64 * 64 * 64,
            out_features=1024,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
        )
        self.kan2 = KANLinear(
            in_features=1024,
            out_features=64 * 16 * 16,  # Match feature map size after pooling
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
        )

        # Final layers to restore input size
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # Upsample to 256x256
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional Layers
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)

        # Self-Attention Block
        attention_weights = self.attention(x)
        x = x * attention_weights.unsqueeze(-1).unsqueeze(-1)

        # Fully Connected Layers
        x = self.flatten(x)
        x = F.selu(self.kan1(x))
        x = self.kan2(x)

        # Reshape back to feature map
        x = x.view(batch_size, 64, 16, 16)  # Match the downsampled size

        # Upsample to original resolution and apply final convolution
        x = self.upsample(x)
        x = self.final_conv(x)

        return x
