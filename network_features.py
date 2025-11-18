"""
Feature extraction from pre-trained networks (VGG16 and Discriminator).

Provides clean interfaces for extracting features from specific layers,
useful for perceptual loss, style transfer, and DeepDream-style synthesis.
"""
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import List, Tuple, Optional
from collections import OrderedDict
import operator


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def normalize_features(features: torch.Tensor, mode: str = 'none') -> torch.Tensor:
    """
    Normalize features using different strategies.

    Args:
        features: Feature tensor to normalize
        mode: Normalization mode ('none', 'numel', 'sqrt')
            - 'none': No normalization
            - 'numel': Divide by total number of elements
            - 'sqrt': Divide by square root of number of elements

    Returns:
        Normalized feature tensor
    """
    if mode == 'numel':
        return features / torch.numel(features)
    elif mode == 'sqrt':
        return features / torch.tensor(torch.numel(features), dtype=torch.float).sqrt()
    else:
        return features


def extract_layer_features(features_dict: OrderedDict, layers: List[str],
                          channels: Optional[List[int]] = None,
                          norm_mode: str = 'none') -> List[torch.Tensor]:
    """
    Extract specific layers and optionally specific channels from a features dictionary.

    Args:
        features_dict: Ordered dictionary of layer_name -> features
        layers: List of layer names to extract
        channels: Optional list of channel indices to extract (None = all channels)
        norm_mode: Normalization mode ('none', 'numel', 'sqrt')

    Returns:
        List of feature tensors for requested layers
    """
    result = []
    for layer in layers:
        feats = features_dict[layer]

        # Channel selection if specified
        if channels is not None:
            max_channels = feats.shape[1]
            valid_channels = [c for c in channels if 0 <= c < max_channels]
            valid_channels = list(set(valid_channels))  # Remove duplicates

            if valid_channels:
                if feats.dim() == 4:  # Conv layers: [B, C, H, W]
                    feats = feats[:, valid_channels, :, :]
                elif feats.dim() == 2:  # FC layers: [B, C]
                    feats = feats[:, valid_channels]

        # Normalization
        feats = normalize_features(feats, norm_mode)
        result.append(feats)

    return result


# ----------------------------------------------------------------------------
# VGG16 Feature Extractors
# ----------------------------------------------------------------------------


class VGG16Features(nn.Module):
    """
    VGG16 feature extractor using PyTorch's pre-trained model.

    Extracts features from specific layers useful for perceptual loss and style transfer.
    Based on Image2StyleGAN (https://arxiv.org/abs/1904.03189).
    """

    def __init__(self, device: torch.device, use_relu: bool = False):
        super().__init__()
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=True).to(device)
        self.vgg16_features = vgg16.features

        # Layer indices: [conv1_1, conv1_2, conv3_2, conv4_2]
        # After ReLU if use_relu=True
        layers = [0, 2, 12, 19]
        if use_relu:
            layers = [l + 1 for l in layers]

        # Create sequential blocks for each feature layer
        self.conv1_1 = self._make_block(0, layers[0] + 1)
        self.conv1_2 = self._make_block(layers[0] + 1, layers[1] + 1)
        self.conv3_2 = self._make_block(layers[1] + 1, layers[2] + 1)
        self.conv4_2 = self._make_block(layers[2] + 1, layers[3] + 1)

        # Freeze parameters
        self.requires_grad_(False)

    def _make_block(self, start: int, end: int) -> nn.Sequential:
        """Create a sequential block from VGG16 features."""
        block = nn.Sequential()
        for i in range(start, end):
            block.add_module(str(i), self.vgg16_features[i])
        return block

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Extract features from all layers."""
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        conv3_2 = self.conv3_2(conv1_2)
        conv4_2 = self.conv4_2(conv3_2)

        # Normalize by number of elements
        conv1_1 = normalize_features(conv1_1, 'numel')
        conv1_2 = normalize_features(conv1_2, 'numel')
        conv3_2 = normalize_features(conv3_2, 'numel')
        conv4_2 = normalize_features(conv4_2, 'numel')

        return conv1_1, conv1_2, conv3_2, conv4_2


class VGG16FeaturesNVIDIA(nn.Module):
    """
    VGG16 feature extractor using NVIDIA's implementation.

    Provides more flexibility in layer selection and normalization.
    ReLU is already included in each conv layer output.
    """

    def __init__(self, vgg16):
        super().__init__()
        # Extract all layers from NVIDIA's VGG16
        self.conv1_1 = vgg16.layers.conv1
        self.conv1_2 = vgg16.layers.conv2
        self.pool1 = vgg16.layers.pool1

        self.conv2_1 = vgg16.layers.conv3
        self.conv2_2 = vgg16.layers.conv4
        self.pool2 = vgg16.layers.pool2

        self.conv3_1 = vgg16.layers.conv5
        self.conv3_2 = vgg16.layers.conv6
        self.conv3_3 = vgg16.layers.conv7
        self.pool3 = vgg16.layers.pool3

        self.conv4_1 = vgg16.layers.conv8
        self.conv4_2 = vgg16.layers.conv9
        self.conv4_3 = vgg16.layers.conv10
        self.pool4 = vgg16.layers.pool4

        self.conv5_1 = vgg16.layers.conv11
        self.conv5_2 = vgg16.layers.conv12
        self.conv5_3 = vgg16.layers.conv13
        self.pool5 = vgg16.layers.pool5

        self.adavgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = vgg16.layers.fc1
        self.fc2 = vgg16.layers.fc2
        self.fc3 = vgg16.layers.fc3
        self.softmax = vgg16.layers.softmax

    def get_layers_features(self, x: torch.Tensor, layers: List[str],
                          normed: bool = False, sqrt_normed: bool = False) -> List[torch.Tensor]:
        """
        Extract features from specified layers.

        Args:
            x: Input image tensor [B, 3, H, W]
            layers: List of layer names to extract
            normed: If True, divide by number of elements
            sqrt_normed: If True, divide by sqrt of number of elements

        Returns:
            List of feature tensors
        """
        # Determine normalization mode
        norm_mode = 'numel' if normed else ('sqrt' if sqrt_normed else 'none')

        # Build features dictionary
        features = OrderedDict()
        features['conv1_1'] = self.conv1_1(x)
        features['conv1_2'] = self.conv1_2(features['conv1_1'])
        features['pool1'] = self.pool1(features['conv1_2'])

        features['conv2_1'] = self.conv2_1(features['pool1'])
        features['conv2_2'] = self.conv2_2(features['conv2_1'])
        features['pool2'] = self.pool2(features['conv2_2'])

        features['conv3_1'] = self.conv3_1(features['pool2'])
        features['conv3_2'] = self.conv3_2(features['conv3_1'])
        features['conv3_3'] = self.conv3_3(features['conv3_2'])
        features['pool3'] = self.pool3(features['conv3_3'])

        features['conv4_1'] = self.conv4_1(features['pool3'])
        features['conv4_2'] = self.conv4_2(features['conv4_1'])
        features['conv4_3'] = self.conv4_3(features['conv4_2'])
        features['pool4'] = self.pool4(features['conv4_3'])

        features['conv5_1'] = self.conv5_1(features['pool4'])
        features['conv5_2'] = self.conv5_2(features['conv5_1'])
        features['conv5_3'] = self.conv5_3(features['conv5_2'])
        features['pool5'] = self.pool5(features['conv5_3'])

        features['adavgpool'] = self.adavgpool(features['pool5'])
        features['fc1'] = self.fc1(features['adavgpool'])
        features['fc2'] = self.fc2(features['fc1'])
        features['fc3'] = self.softmax(self.fc3(features['fc2']))

        return extract_layer_features(features, layers, norm_mode=norm_mode)


# ----------------------------------------------------------------------------
# Discriminator Feature Extractor
# ----------------------------------------------------------------------------


class DiscriminatorFeatures(nn.Module):
    """
    Feature extractor for StyleGAN2/3 Discriminator.

    Extracts intermediate features from discriminator layers, useful for
    DeepDream-style synthesis and perceptual analysis.
    """

    def __init__(self, D):
        super().__init__()
        self.block_resolutions = D.block_resolutions

        # Extract all discriminator blocks
        for res in self.block_resolutions:
            if res == D.img_resolution:
                setattr(self, 'from_rgb', operator.attrgetter(f'b{res}.fromrgb')(D))
            setattr(self, f'b{res}_skip', operator.attrgetter(f'b{res}.skip')(D))
            setattr(self, f'b{res}_conv0', operator.attrgetter(f'b{res}.conv0')(D))
            setattr(self, f'b{res}_conv1', operator.attrgetter(f'b{res}.conv1')(D))

        # Final block (resolution 4x4)
        self.b4_mbstd = D.b4.mbstd
        self.b4_conv = D.b4.conv
        self.adavgpool = nn.AdaptiveAvgPool2d(4)  # Handle different input sizes
        self.fc = D.b4.fc
        self.out = D.b4.out

    def get_block_resolutions(self) -> List[int]:
        """Get available block resolutions."""
        return self.block_resolutions

    def get_layers_features(self, x: torch.Tensor, layers: Optional[List[str]] = None,
                          channels: Optional[List[int]] = None,
                          normed: bool = False, sqrt_normed: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Extract features from specified discriminator layers.

        Args:
            x: Input image tensor [B, 3, H, W]
            layers: List of layer names (None = ['out'])
            channels: List of channel indices to extract (None = all)
            normed: If True, divide by number of elements
            sqrt_normed: If True, divide by sqrt of number of elements

        Returns:
            Tuple of feature tensors

        Available layers:
            - 'from_rgb': Initial RGB conversion
            - 'b{res}_conv0', 'b{res}_conv1': Convolution layers at each resolution
            - 'b{res}_skip': Skip connections
            - 'b4_mbstd', 'b4_conv': Final block
            - 'fc', 'out': Fully connected and output layers
        """
        assert not (normed and sqrt_normed), 'Choose only one normalization mode!'

        layers = layers if layers is not None else ['out']
        norm_mode = 'numel' if normed else ('sqrt' if sqrt_normed else 'none')

        # Build features dictionary
        features = OrderedDict()
        features['from_rgb'] = getattr(self, 'from_rgb')(x)

        # Process each resolution block
        for idx, res in enumerate(self.block_resolutions):
            prev_layer = 'from_rgb' if idx == 0 else f'b{self.block_resolutions[idx-1]}_conv1'

            features[f'b{res}_skip'] = getattr(self, f'b{res}_skip')(features[prev_layer], gain=np.sqrt(0.5))
            features[f'b{res}_conv0'] = getattr(self, f'b{res}_conv0')(features[prev_layer])
            features[f'b{res}_conv1'] = getattr(self, f'b{res}_conv1')(features[f'b{res}_conv0'], gain=np.sqrt(0.5))

            # Skip connection addition (in-place)
            features[f'b{res}_conv1'] = features[f'b{res}_skip'].add_(features[f'b{res}_conv1'])

        # Final block (always at 8x8 → 4x4)
        features['b4_mbstd'] = self.b4_mbstd(features['b8_conv1'])
        features['b4_conv'] = self.b4_conv(features['b4_mbstd'])
        features['b4_conv'] = self.adavgpool(features['b4_conv'])
        features['fc'] = self.fc(features['b4_conv'].flatten(1))
        features['out'] = self.out(features['fc'])

        return tuple(extract_layer_features(features, layers, channels, norm_mode))
