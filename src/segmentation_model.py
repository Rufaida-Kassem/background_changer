"""DeepLabV3 model construction for binary person segmentation."""

from __future__ import annotations

from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class SegmentationModelFactory:
    """Create DeepLabV3 with one output channel for binary segmentation."""

    @staticmethod
    def create(pretrained: bool = True) -> nn.Module:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights, weights_backbone=None)

        classifier = model.classifier
        classifier[-1] = nn.Conv2d(
            classifier[-1].in_channels,
            1,
            kernel_size=1,
        )
        if model.aux_classifier is not None:
            auxiliary_classifier = model.aux_classifier
            auxiliary_classifier[-1] = nn.Conv2d(
                auxiliary_classifier[-1].in_channels,
                1,
                kernel_size=1,
            )
        return model
