"""Configurable convolutional neural network for Stage 3 image datasets."""

import torch
from torch import nn

from local_code.base_class.method import method


class Method_CNN_Stage3(method, nn.Module):
    """CNN with configurable convolution widths, kernels, and dense layers."""

    def __init__(
        self,
        input_channels,
        number_of_classes,
        conv_channels=(32, 64),
        kernel_sizes=(3,),
        hidden_dims=(128,),
        dropout=0.0,
        adaptive_pool_size=4,
    ):
        method.__init__(self, 'Stage 3 CNN', 'Configurable image classifier')
        nn.Module.__init__(self)

        conv_channels = tuple(conv_channels)
        kernel_sizes = tuple(kernel_sizes)
        hidden_dims = tuple(hidden_dims)
        if not conv_channels:
            raise ValueError('conv_channels must contain at least one layer')
        if len(kernel_sizes) == 1:
            kernel_sizes = kernel_sizes * len(conv_channels)
        if len(kernel_sizes) != len(conv_channels):
            raise ValueError('provide one kernel size or one per convolution layer')
        if any(size <= 0 or size % 2 == 0 for size in kernel_sizes):
            raise ValueError('kernel sizes must be positive odd integers')
        if adaptive_pool_size <= 0:
            raise ValueError('adaptive_pool_size must be positive')

        layers = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                # ceil_mode keeps very small inputs valid in deeper ablations.
                nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            ])
            in_channels = out_channels

        self.feature_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))

        classifier_layers = []
        layer_input_size = conv_channels[-1] * adaptive_pool_size * adaptive_pool_size
        for hidden_size in hidden_dims:
            classifier_layers.extend([nn.Linear(layer_input_size, hidden_size), nn.ReLU()])
            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))
            layer_input_size = hidden_size
        classifier_layers.append(nn.Linear(layer_input_size, number_of_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
