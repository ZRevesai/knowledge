"""
MobileNetV3 Architecture with Custom Modifications

Implements:
1. Depthwise Separable Convolutions
2. Inverted Residuals with Linear Bottlenecks
3. Integration with Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SqueezeExcitation, ShuffleAttention


class HSwish(nn.Module):
    """Hard Swish activation function."""
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.

    Applies a 3x3 convolution on each channel separately (depthwise),
    followed by a 1x1 convolution to project output channels (pointwise).

    Reference: Figure 2 (STAGE A) in the paper

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
        activation: Activation function
    """
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise: 3x3 conv on each channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            activation(inplace=True)
        )

        # Pointwise: 1x1 conv to project channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block with Linear Bottleneck.

    Reference: Figure 2 (STAGE B) in the paper

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
        expand_ratio: Expansion ratio for hidden dimension (default: 6)
        reduction_ratio: Reduction ratio for SE block (default: 4)
        use_se: Whether to use Squeeze-Excitation
        use_sa: Whether to use Shuffle Attention
        activation: Activation function
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6,
                 reduction_ratio=4, use_se=False, use_sa=False,
                 activation=nn.ReLU):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Reduce channels based on reduction ratio
        hidden_dim = in_channels * expand_ratio // reduction_ratio

        layers = []

        # Expansion phase (only if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels,
                     hidden_dim if expand_ratio != 1 else in_channels,
                     kernel_size=3, stride=stride, padding=1,
                     groups=hidden_dim if expand_ratio != 1 else in_channels,
                     bias=False),
            nn.BatchNorm2d(hidden_dim if expand_ratio != 1 else in_channels),
            activation(inplace=True)
        ])

        self.conv = nn.Sequential(*layers)

        # Attention mechanisms
        self.use_se = use_se
        self.use_sa = use_sa

        if use_se:
            self.se = SqueezeExcitation(
                hidden_dim if expand_ratio != 1 else in_channels,
                reduction_ratio=reduction_ratio
            )

        if use_sa:
            # Ensure channels are compatible with Shuffle Attention
            sa_channels = hidden_dim if expand_ratio != 1 else in_channels
            if sa_channels % 16 == 0:  # Must be divisible by 2*groups (2*8=16)
                self.sa = ShuffleAttention(sa_channels, groups=8)
            else:
                self.use_sa = False  # Disable if not compatible

        # Projection phase (linear bottleneck)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels,
                     out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv(x)

        # Apply attention mechanisms
        if self.use_se:
            out = self.se(out)

        if self.use_sa:
            out = self.sa(out)

        out = self.project(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        return out


class MobileNetV3(nn.Module):
    """
    Modified MobileNetV3 for Nutrient Analysis.

    Architecture features:
    - Input: 224x224x3
    - 5 convolutional stages
    - Progressive channel increase: 32 -> 320
    - Depthwise separable convolutions throughout
    - Inverted residuals with linear bottlenecks
    - Integrated attention mechanisms

    Args:
        num_classes: Number of food categories
        width_multiplier: Width multiplier for channels
        use_se: Whether to use Squeeze-Excitation blocks
        use_sa: Whether to use Shuffle Attention
        dropout: Dropout rate
    """
    def __init__(self, num_classes=500, width_multiplier=1.0,
                 use_se=True, use_sa=True, dropout=0.2):
        super(MobileNetV3, self).__init__()

        def _make_divisible(v, divisor=8):
            """Ensure channels are divisible by divisor."""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Initial convolution
        input_channels = _make_divisible(32 * width_multiplier)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            HSwish()
        )

        # MobileNetV3 configuration
        # [expansion, out_channels, num_blocks, stride, use_se, use_sa, activation]
        config = [
            # Stage 1
            [1, 16, 1, 1, False, False, nn.ReLU],
            # Stage 2
            [6, 24, 2, 2, False, False, nn.ReLU],
            # Stage 3
            [6, 40, 3, 2, True, False, nn.ReLU],
            # Stage 4
            [6, 80, 4, 2, True, True, HSwish],
            # Stage 5
            [6, 160, 3, 1, True, True, HSwish],
            [6, 320, 1, 1, True, True, HSwish],
        ]

        # Build inverted residual blocks
        layers = []
        for exp_ratio, out_ch, num_blocks, stride, se, sa, act in config:
            out_channels = _make_divisible(out_ch * width_multiplier)

            for i in range(num_blocks):
                layers.append(
                    InvertedResidual(
                        in_channels=input_channels,
                        out_channels=out_channels,
                        stride=stride if i == 0 else 1,
                        expand_ratio=exp_ratio,
                        reduction_ratio=4,
                        use_se=se and use_se,
                        use_sa=sa and use_sa,
                        activation=act
                    )
                )
                input_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Final convolution
        final_channels = _make_divisible(1280 * width_multiplier)
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            HSwish()
        )

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Classifier (will be replaced by multi-task heads in NutrientAnalysisModel)
        self.classifier = nn.Linear(final_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            features: Feature tensor for multi-task heads
            logits: Classification logits (if using standalone)
        """
        x = self.conv_stem(x)
        x = self.features(x)
        x = self.conv_head(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.dropout(features)
        logits = self.classifier(x)

        return features, logits


if __name__ == "__main__":
    # Test MobileNetV3
    model = MobileNetV3(num_classes=500, width_multiplier=1.0,
                        use_se=True, use_sa=True, dropout=0.2)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    features, logits = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")

    print("\nMobileNetV3 model created successfully!")
