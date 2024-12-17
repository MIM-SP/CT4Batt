import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class Attention2D(nn.Module):
    def __init__(self, in_channels, ratio=0.25, num_kernels=4, temperature=5, kernel_size=3, activation_fn=nn.ReLU, init_weight=True):
        super(Attention2D, self).__init__()
        assert temperature >= 1, "Temperature must be >= 1."
        
        self.temperature = temperature
        hidden_channels = max(int(in_channels * ratio), 1)
        self.activation_fn = activation_fn()

        # Attention layers
        self.global_pool = nn.AdaptiveAvgPool2d(7)  # Ensure 7x7 output
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_channels, num_kernels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.norm = nn.LayerNorm(num_kernels)  # Apply LayerNorm to the last dimension dynamically

        # ROI prediction head: Predict 4 bbox coordinates
        self.roi_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Reduce spatial dimensions to 1x1
            nn.Conv2d(hidden_channels, 4, kernel_size=1)  # Output 4 bbox coordinates (x1, y1, x2, y2)
        )

        # Initialize weights
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Global context extraction
        context = self.global_pool(x)  # [Batch, Channels, 7, 7]
        
        # Attention weights computation
        attention = self.activation_fn(self.fc1(context))  # [Batch, Hidden Channels, 7, 7]
        attention = self.fc2(attention).view(batch_size, -1)  # [Batch, num_kernels]

        # Reshape attention if necessary
        if attention.shape[-1] != 4:
            attention = attention.view(batch_size, -1, 4)
        
        # Normalize attention weights using softmax with temperature
        attention = self.norm(attention)  # Apply LayerNorm
        attention_weights = F.softmax(attention / (self.temperature + 1e-5), dim=1)  # Smoothed softmax
        
        # ROI prediction (4 bbox values)
        roi_bbox = self.roi_head(context).view(batch_size, 4)

        return attention_weights, roi_bbox

class CNN2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=11, ratio=0.25, stride=1, dilation=1, groups=1, bias=True, num_kernels=4, temperature=5, activation_fn=nn.Softplus, init_weight=True, roi_size=(7, 7)):
        super(CNN2D, self).__init__()
        assert in_channels % groups == 0, "Input channels must be divisible by groups."
        self.iteration_count = 0  # Initialize iteration counter
        self.init_iterations = 10  # Number of iterations to apply random init
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.roi_size = roi_size

        self.attention = Attention2D(in_channels, ratio, num_kernels, temperature, kernel_size, activation_fn)

        # Kernel weights and bias
        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels)) if bias else None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Resize input to ensure compatibility with AdaptiveAvgPool2d
        target_size = (7 * (height // 7), 7 * (width // 7))  # Make divisible by 7
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # Compute attention weights and ROI
        attention_weights, roi_bbox = self.attention(x)

        # Convert normalized ROI to pixel coordinates
        roi_boxes = self._convert_to_roi_boxes(roi_bbox, target_size[0], target_size[1], batch_size)

        # Perform ROI Align
        zoomed_regions = roi_align(x, roi_boxes, self.roi_size)

        # Resize zoomed regions to 256x256
        x_zoomed = F.interpolate(zoomed_regions, size=(256, 256), mode='bilinear', align_corners=False)

        # Flatten kernel weights
        weight_flat = self.weight.view(self.num_kernels, -1)

        # Reshape attention_weights to match expected shape
        attention_weights = attention_weights.view(batch_size, self.num_kernels, -1).sum(dim=2)

        # Aggregate weights
        aggregated_weight = torch.matmul(attention_weights, weight_flat).view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        # Bias aggregation
        aggregated_bias = None
        if self.bias is not None:
            aggregated_bias = torch.matmul(attention_weights, self.bias).view(-1)

        # Set groups to 1
        adjusted_groups = 1

        # Apply convolution
        x_zoomed = x_zoomed.repeat(1, self.in_channels, 1, 1)  # Match channel count
        output = F.conv2d(
            x_zoomed,
            weight=aggregated_weight,
            bias=aggregated_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=adjusted_groups
        )
        attention_weights = torch.softmax(output.mean(dim=(2, 3), keepdim=True), dim=1)  # Spatial mean + softmax
        output = torch.sum(output * attention_weights, dim=1, keepdim=True)  # Weighted sum

        # Reshape back
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))

        # Apply random initialization only for the first 'init_iterations'
        if self.iteration_count < self.init_iterations:
            random_init = torch.empty_like(output).uniform_(-1, 1).to(output.device)
            output = output + random_init  # Add random initialization
        
        output = torch.tanh(output)  # Ensure range [-1, 1]

        # Update the iteration counter
        self.iteration_count += 1

        return output

    def _convert_to_roi_boxes(self, roi_bbox, height, width, batch_size):
        """
        Convert normalized ROI bbox to absolute coordinates for ROI Align.
        """
        roi_boxes = []
        for i in range(batch_size):
            x1 = roi_bbox[i, 0] * width
            y1 = roi_bbox[i, 1] * height
            x2 = roi_bbox[i, 2] * width
            y2 = roi_bbox[i, 3] * height
            roi_boxes.append([i, x1, y1, x2, y2])
        return torch.tensor(roi_boxes, device=roi_bbox.device, dtype=torch.float32)