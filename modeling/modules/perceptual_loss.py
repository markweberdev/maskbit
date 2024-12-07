"""This file contains the definition of the perceptual loss."""

import torch

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "resnet50", compute_perceptual_loss_on_logits: bool = True):
        """ Initialize the perceptual loss.

        Args:
            model_name -> str: The name of the model to use.
            compute_perceptual_loss_on_logits -> bool: Whether to compute the perceptual loss on the logits 
                or the features.
        """
        super().__init__()
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return_nodes = {"layer4": "features", "fc": "logits"}
        elif model_name == "convnext_s":
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            return_nodes = {"features": "features", "classifier": "logits"}

        if compute_perceptual_loss_on_logits:
            self.model = model
        else:
            self.model = create_feature_extractor(model, return_nodes=return_nodes)

        self.compute_perceptual_loss_on_logits = compute_perceptual_loss_on_logits

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the perceptual loss.

        Args:
            input -> torch.Tensor: The input tensor.
            target -> torch.Tensor: The target tensor.

        Returns:
            loss -> torch.Tensor: The perceptual loss.
        """
        input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", antialias=True, align_corners=False)
        target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", antialias=True, align_corners=False)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        features_input = self.model(input)
        features_target = self.model(target)

        if self.compute_perceptual_loss_on_logits:
            loss = torch.nn.functional.mse_loss(features_input, features_target, reduction="mean")
        else:
            loss = torch.nn.functional.mse_loss(features_input["features"], features_target["features"], reduction="mean")
            loss += torch.nn.functional.mse_loss(features_input["logits"], features_target["logits"], reduction="mean")
        return loss


if  __name__ == "__main__":
    model = PerceptualLoss()
    input = torch.randn(2,3,256,256).clamp_(0, 1)
    target = torch.randn(2,3,256,256).clamp_(0, 1)
    loss = model(input, target)
    print(loss)