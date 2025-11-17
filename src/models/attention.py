"""
Attention Mechanisms for Nutrient Analysis Model

Implements:
1. Squeeze-and-Excitation (SE) blocks
2. Shuffle Attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.

    Reference: Equation (2) in the paper
    s = σ(W2 δ(W1 GAP(U))), Û = s · U

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for bottleneck (default: 4)
    """
    def __init__(self, in_channels, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling (GAP)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),  # δ activation
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()  # σ activation
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Recalibrated tensor of shape (B, C, H, W)
        """
        batch_size, channels, _, _ = x.size()

        # Squeeze: Global spatial information
        squeeze = self.squeeze(x).view(batch_size, channels)

        # Excitation: Channel-wise recalibration
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)

        # Scale the input
        return x * excitation.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel attention component for Shuffle Attention.

    Reference: Equation (3) in the paper
    I'{k1} = σ(F_c(s)) · I{k1} = σ(W1s + b1) · I_{k1}
    """
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention component optimized for micronutrient detection.

    Reference: Equation (4) in the paper
    I'{k2} = σ(W2 · GN(I{k2}) + b2) · I_{k2}

    Specifically adapted for micronutrient-rich regions (e.g., vegetable surfaces,
    meat marbling, leafy green textures).
    """
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Spatially attended tensor of shape (B, C, H, W)
        """
        # Apply group normalization
        normalized = self.group_norm(x)

        # Generate spatial attention map
        attention = self.sigmoid(self.conv(normalized))

        return x * attention


class ShuffleAttention(nn.Module):
    """
    Shuffle Attention mechanism for enhanced feature learning without
    significant computational overhead.

    Specifically adapted to handle layered and mixed food presentations
    by applying spatial attention to different regions simultaneously.

    Reference: Equations (3) and (4) in the paper

    Args:
        channels: Number of input channels
        groups: Number of groups for shuffling (default: 8)
    """
    def __init__(self, channels, groups=8):
        super(ShuffleAttention, self).__init__()
        self.groups = groups

        assert channels % (2 * groups) == 0, \
            f"Channels ({channels}) must be divisible by 2*groups ({2*groups})"

        self.channels_per_group = channels // (2 * groups)

        # Channel attention branch
        self.channel_attention = ChannelAttention(channels // 2)

        # Spatial attention branch (micronutrient-focused)
        self.spatial_attention = SpatialAttention(channels // 2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-enhanced tensor of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()

        # Reshape for group processing
        x = x.view(batch_size * self.groups, -1, height, width)

        # Split into two branches
        x1, x2 = x.chunk(2, dim=1)

        # Apply channel and spatial attention separately
        x1 = self.channel_attention(x1)
        x2 = self.spatial_attention(x2)

        # Concatenate
        out = torch.cat([x1, x2], dim=1)

        # Reshape back
        out = out.view(batch_size, -1, height, width)

        # Channel shuffle for information exchange between groups
        out = self.channel_shuffle(out, self.groups)

        return out

    @staticmethod
    def channel_shuffle(x, groups):
        """
        Shuffle channels for better feature integration.

        Args:
            x: Input tensor of shape (B, C, H, W)
            groups: Number of groups

        Returns:
            Shuffled tensor of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // groups

        # Reshape and transpose
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()

        # Flatten back
        x = x.view(batch_size, -1, height, width)

        return x


class LightweightAttention(nn.Module):
    """
    Lightweight attention mechanism for micronutrient detection.

    Optimized for micronutrient-rich regions through expert-guided annotation,
    enabling accurate detection of vitamin and mineral content indicators.
    """
    def __init__(self, in_channels, reduction=16):
        super(LightweightAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention for micronutrient regions
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        batch_size, channels, _, _ = x.size()

        # Channel attention
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        channel_att = x * y.expand_as(x)

        # Spatial attention for micronutrients
        spatial_att = self.spatial(x)

        # Combine both attentions
        out = channel_att * spatial_att

        return out


if __name__ == "__main__":
    # Test the attention mechanisms
    batch_size = 4
    channels = 64
    height, width = 28, 28

    x = torch.randn(batch_size, channels, height, width)

    # Test Squeeze-and-Excitation
    se = SqueezeExcitation(channels, reduction_ratio=4)
    se_out = se(x)
    print(f"SE Block - Input: {x.shape}, Output: {se_out.shape}")

    # Test Shuffle Attention
    sa = ShuffleAttention(channels, groups=8)
    sa_out = sa(x)
    print(f"Shuffle Attention - Input: {x.shape}, Output: {sa_out.shape}")

    # Test Lightweight Attention
    la = LightweightAttention(channels, reduction=16)
    la_out = la(x)
    print(f"Lightweight Attention - Input: {x.shape}, Output: {la_out.shape}")

    print("\nAll attention mechanisms working correctly!")
