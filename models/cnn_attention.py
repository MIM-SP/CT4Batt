import torch
import torch.nn as nn
import torch.nn.functional as F

class attention2D(nn.Module):
    """
    Attention mechanism for 2D dynamic convolution.
    """
    def __init__(self, in_channels, ratio=0.25, num_kernels=4, temperature=34, init_weight=True):
        super(attention2D, self).__init__()
        assert temperature % 3 == 1, "Temperature must satisfy `temperature % 3 == 1`."
        
        self.temperature = temperature
        hidden_channels = max(int(in_channels * ratio), 1)

        # Attention layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_channels, num_kernels, kernel_size=1, bias=True)

        # Initialize weights
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def update_temperature(self):
        """Reduce the temperature to sharpen the attention distribution."""
        if self.temperature > 1:
            self.temperature -= 3
            print(f"Updated temperature: {self.temperature}")

    def forward(self, x):
        # Global context extraction
        context = self.global_pool(x)
        
        # Fully connected layers
        attention = F.relu(self.fc1(context))
        attention = self.fc2(attention).view(x.size(0), -1)  # (Batch, num_kernels)

        # Normalize with softmax and temperature scaling
        return F.softmax(attention / self.temperature, dim=1)


class cnn2D(nn.Module):
    """
    Dynamic 2D Convolution with attention-based kernel selection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, num_kernels=4, temperature=34, init_weight=True):
        super(cnn2D, self).__init__()
        assert in_channels % groups == 0, "Input channels must be divisible by groups."
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels

        # Attention mechanism
        self.attention = attention2D(in_channels, ratio, num_kernels, temperature)

        # Kernel weights and bias
        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size), 
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels)) if bias else None

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
        return output
