from typing import Mapping, Text, Tuple
import torch
import torch.nn.functional as F


from .lpips import LPIPS
from .perceptual_loss import PerceptualLoss
from . import gan_utils


def create_perception_loss(perception_loss: str, compute_on_logits: bool = True) -> torch.nn.Module:
    """ Creates the perception loss.

    Args:
        perception_loss -> str: The name of the perception loss.
        compute_on_logits -> bool: Whether to compute the loss on logits or on multiple features.

    Returns:
        perception_loss -> torch.nn.Module: The perception loss.
    """
    if perception_loss == "lpips":
        return LPIPS().eval()
    elif perception_loss in ("resnet50", "convnext_s"):
        return PerceptualLoss(
            model_name=perception_loss,
            compute_perceptual_loss_on_logits=compute_on_logits
        ).eval()
    else:
        raise ValueError(f"Perception loss {perception_loss} is not supported.")


class VQGANLoss(torch.nn.Module):
    def __init__(
        self,
        discriminator_config,
        loss_config,
    ):
        """ Initializes the VQGAN loss.

        Args:
            discriminator_config: The configuration of the discriminator.
            loss_config: The configuration of the loss.
        """
        super().__init__()
        assert loss_config.discriminator_loss in ("hinge", "vanilla", "non-saturating")
        assert loss_config.reconstruction_loss in ("l2", "l1")
        assert loss_config.discriminator_gradient_penalty in ("none", "adopt_weight")

        self.discriminator = gan_utils.create_discriminator(discriminator_config)

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.get("reconstruction_weight", 1.0)
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = create_perception_loss(loss_config.perceptual_loss, loss_config.get("perceptual_loss_on_logits", True))
        self.perceptual_weight = loss_config.perceptual_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.ema_decay = loss_config.get("ema_decay", 0.999)

        self.entropy_annealing_steps = loss_config.get("entropy_annealing_steps", 2000)
        self.entropy_annealing_factor = loss_config.get("entropy_annealing_factor", 0.0)

        self.discriminator_iter_start = loss_config.discriminator_start

        if loss_config.discriminator_loss == "hinge":
            self.discriminator_loss = gan_utils.hinge_d_loss
        elif loss_config.discriminator_loss == "vanilla":
            self.discriminator_loss = gan_utils.vanilla_d_loss
        elif loss_config.discriminator_loss == "non-saturating":
            self.discriminator_loss = gan_utils.non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{loss_config.discriminator_loss}'.")
    
        if loss_config.discriminator_loss == "hinge":
            self.generator_loss = gan_utils.hinge_g_loss
        elif loss_config.discriminator_loss == "vanilla":
            self.generator_loss = gan_utils.hinge_g_loss
        elif loss_config.discriminator_loss == "non-saturating":
            self.generator_loss = gan_utils.non_saturating_g_loss
        else:
            raise ValueError(f"Unknown GAN loss '{loss_config.discriminator_loss}'.")

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight

        self.discriminator_gradient_penalty = (
            "" if loss_config.discriminator_gradient_penalty == "none" else loss_config.discriminator_gradient_penalty
        )
        self.discriminator_penalty_cost = loss_config.discriminator_penalty_cost

        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))


    def calculate_adaptive_weight(self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer) -> torch.Tensor:
        """ Calculates the adaptive weight for the discriminator loss.

        Args:
            nll_loss -> torch.Tensor: The NLL loss.
            g_loss -> torch.Tensor: The generator loss.
            last_layer: The last layer of the model.

        Returns:
            d_weight -> torch.Tensor: The adaptive weight for the discriminator loss.
        """
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight


    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        last_layer,
        mode: str = "gen",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the VQGAN loss for the generator or discriminator.

        Args:
            inputs -> torch.Tensor: The input images.
            reconstructions -> torch.Tensor: The reconstructed images.
            extra_result_dict -> Mapping[Text, torch.Tensor]: The extra result dictionary.
            global_step -> int: The global step.
            last_layer: The last layer of the model.
            mode -> str: The mode. Must be either "gen" or "disc".
        
        Returns:
            loss -> torch.Tensor: The loss.
            loss_dict -> Mapping[Text, torch.Tensor]: The loss dictionary for logging individual losses.
        """
        assert mode in ("gen", "disc")
        if mode == "gen":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step, last_layer)
        elif mode == "disc":
            return self._forward_discriminator(inputs, reconstructions, extra_result_dict, global_step)

        
    def should_discriminator_be_trained(self, global_step : int):
        """Returns if the discriminator should be trained at given step."""
        return global_step >= self.discriminator_iter_start

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        last_layer
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the VQGAN loss for the generator.
        
        Args:
            inputs -> torch.Tensor: The input images.
            reconstructions -> torch.Tensor: The reconstructed images.
            extra_result_dict -> Mapping[Text, torch.Tensor]: The extra result dictionary.
            global_step -> int: The global step.
            last_layer: The last layer of the model.
            
        Returns:
            loss -> torch.Tensor: The loss.
            loss_dict -> Mapping[Text, torch.Tensor]: The loss dictionary for logging individual losses.
        """
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        else:
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        reconstruction_loss *= self.reconstruction_weight

        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        generator_loss = torch.zeros((), device=inputs.device)
        extra_generator_loss = torch.zeros((), device=inputs.device)

        discriminator_factor = gan_utils.adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)

        d_weight = 1.0
        if discriminator_factor > 0.0:
            # Disable discriminator gradients
            gan_utils.toggle_off_gradients(self.discriminator)

            logits_fake = self.discriminator(reconstructions)
            generator_loss = self.generator_loss(logits_fake)

            if self.discriminator_gradient_penalty == "adopt_weight":
                d_weight *= self.calculate_adaptive_weight(
                    reconstruction_loss + self.perceptual_weight * perceptual_loss,
                    generator_loss,
                    last_layer=last_layer
                )
        d_weight *= self.discriminator_weight

        quantizer_loss = extra_result_dict["quantizer_loss"]
        if self.entropy_annealing_factor > 0.0:
            quantizer_loss += (
                max(0.0, 1 - global_step / self.entropy_annealing_steps)
                * self.entropy_annealing_factor * extra_result_dict["entropy_loss"]
            )

        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * (generator_loss + extra_generator_loss)
        )

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * (generator_loss + extra_generator_loss)).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            entropy_loss=extra_result_dict["entropy_loss"].detach(),
            per_sample_entropy=extra_result_dict["per_sample_entropy"],
            avg_entropy=extra_result_dict["avg_entropy"],
            d_weight=d_weight,
            gan_loss=generator_loss.detach()
        )
        if "codebook_loss" in extra_result_dict:
            loss_dict["codebook_loss"] = extra_result_dict["codebook_loss"].detach()

        return total_loss, loss_dict

    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the VQGAN loss for the discriminator.

        Args:
            inputs -> torch.Tensor: The input images.
            reconstructions -> torch.Tensor: The reconstructed images.
            extra_result_dict -> Mapping[Text, torch.Tensor]: The extra result dictionary.
            global_step -> int: The global step.

        Returns:
            loss -> torch.Tensor: The loss.
            loss_dict -> Mapping[Text, torch.Tensor]: The loss dictionary for logging individual losses.
        """

        discriminator_factor = gan_utils.adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)
        loss_dict = {}
        # Turn on gradients on
        gan_utils.toggle_on_gradients(self.discriminator)

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * self.discriminator_loss(logits_real=logits_real, logits_fake=logits_fake)

        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = gan_utils.compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.ema_decay + torch.mean(logits_real).detach() * (1 - self.ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.ema_decay + torch.mean(logits_fake).detach() * (1 - self.ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )

        return discriminator_loss, loss_dict


class MLMLoss(torch.nn.Module):
    def __init__(self, label_smoothing: float = 0.1, sum_splits: bool = False):
        """ Initializes the MLM loss, which is essentially a CrossEntropy loss with label smoothing.

        Args:
            label_smoothing -> float: The label smoothing factor.
            sum_splits -> bool: Whether to sum the loss over the splits.
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.sum_splits = sum_splits

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the MLM loss.
        
        Args:
            inputs -> torch.Tensor: The input logits.
            targets -> torch.Tensor: The target tokens.
            masks -> torch.Tensor: The mask for the tokens.

        Returns:
            loss -> torch.Tensor: The loss.
            loss_dict -> Mapping[Text, torch.Tensor]: The loss dictionary for logging individual losses.
        """
        b, n, m, codebook_size = inputs.shape
        loss = self.criterion(inputs.reshape(-1, codebook_size), targets.view(-1))
        
        correct_tokens = (torch.argmax(inputs.detach(), dim=-1) == targets).float().mean() ** m

        masked_input = inputs[masks, :].detach()
        masked_loss = self.criterion(masked_input, targets[masks])
        masked_correct_tokens = (torch.argmax(masked_input, dim=-1) == targets[masks]).float().mean() ** m

        if self.sum_splits:
            loss *= m
            masked_loss *= m

        loss_dict = {
            "mlm_loss": loss,
            "correct_tokens": correct_tokens,
            "masked_token_loss": masked_loss,
            "masked_correct_tokens": masked_correct_tokens,
            }

        return loss, loss_dict


if __name__ == "__main__":
    loss_module = MLMLoss()

    batchsize = 2
    codebook_dim = 4
    num_codebooks = 1

    logits = torch.rand((batchsize, 3, num_codebooks, codebook_dim))
    targets = torch.randint(0, codebook_dim, (batchsize, 3, num_codebooks))
    masks = torch.randint(0, 2, (batchsize, 3, num_codebooks), dtype=bool)

    loss, loss_dict = loss_module(logits, targets, masks)
    print(logits)
    print(targets)
    print(masks)
    print(loss, loss_dict)