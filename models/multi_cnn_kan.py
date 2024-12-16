import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_linear import KANLinear
from cnn_attention import AttentionModule

class CNNWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=64, kernel_size=3, activation_fn=nn.ReLU):
        super(CNNWithAttention, self).__init__()
        
        self.activation_fn = activation_fn()
        
        # Encoder part (downsampling)
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(base, base*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(base*2, base*4, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample
        
        # Attention mechanism (optional)
        self.attention = AttentionModule(base*4, base*2, kernel_size=kernel_size)
        
        # Decoder part (upsampling)
        self.deconv1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.deconv2 = nn.ConvTranspose2d(base*2, base, kernel_size=kernel_size, padding=kernel_size//2)
        self.deconv3 = nn.ConvTranspose2d(base, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Optional: Ensure final output size is the same as target size
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        # Fully connected layers
        self.kan1 = KANLinear(in_features=base*4, out_features=base*2)
        self.kan2 = KANLinear(in_features=base*2, out_features=out_channels)

    def forward(self, x):
        # Encoder part
        x1 = self.activation_fn(self.conv1(x))
        x2 = self.pool(self.activation_fn(self.conv2(x1)))
        x3 = self.pool(self.activation_fn(self.conv3(x2)))
        
        # Attention mechanism
        attention_map = self.attention(x3)
        x3 = x3 * attention_map  # Apply attention to the feature map
        
        # Decoder part (upsampling)
        x4 = self.activation_fn(self.deconv1(x3))
        x5 = self.activation_fn(self.deconv2(x4))
        output = self.deconv3(x5)
        output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)

        # Ensure output is same size as input (256x256)
        output = self.final_conv(output)  # Optional: use final convolution to fine-tune output size
        
        # Apply fully connected layers
        output = output.view(output.size(0), -1)  # Flatten for fully connected layers
        output = self.kan1(output)
        output = self.kan2(output)

        return output