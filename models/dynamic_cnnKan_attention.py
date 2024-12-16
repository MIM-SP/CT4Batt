
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_linear import KANLinear
from dynamic_cnn_attention import Attention2D

class CNN2D(nn.Module):
    """
    Dynamic 2D Convolution with attention-based kernel selection.
    """
    def __init__(self, in_channels, out_channels, base=64, kernel_size=3, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, num_kernels=4, temperature=34, activation_fn=nn.ReLU, init_weight=True):
        super(CNN2D, self).__init__()
        assert in_channels % groups == 0, "Input channels must be divisible by groups."
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.activation_fn = activation_fn()

        # Attention mechanism
        self.attention = Attention2D(in_channels, ratio, num_kernels, temperature, kernel_size, activation_fn)

        # Kernel weights and bias
        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size), 
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels)) if bias else None

        # Fully connected layer
        self.kan = KANLinear(in_features=base*4, out_features=out_channels)

        # Initialize weights
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def update_temperature(self):
        """Update the temperature in the attention mechanism."""
        self.attention.update_temperature()

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Compute attention weights
        attention_weights = self.attention(x)  # (Batch, num_kernels)

        # Flatten kernel weights for dynamic aggregation
        weight_flat = self.weight.view(self.num_kernels, -1)  # (num_kernels, out_channels * in_channels // groups * kernel_size^2)

        # Dynamically aggregate weights and bias
        aggregated_weight = torch.matmul(attention_weights, weight_flat).view(
            batch_size * self.out_channels, 
            self.in_channels // self.groups, 
            self.kernel_size, 
            self.kernel_size
        )
        if self.bias is not None:
            aggregated_bias = torch.matmul(attention_weights, self.bias).view(-1)
        else:
            aggregated_bias = None

        # Perform convolution
        x = x.view(1, -1, height, width)  # Combine batch and channel dimensions for grouped convolution
        output = F.conv2d(
            x, 
            weight=aggregated_weight, 
            bias=aggregated_bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups * batch_size
        )

        # Reshape back to (B, out_channels, H_out, W_out)
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))

        # Apply fully connected layer
        output = output.view(batch_size, -1)  # Flatten for fully connected layer
        output = self.kan(output)
        
        return output