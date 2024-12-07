"""This file contains the definition of the original Taming VQGAN.

We thank the following public implementations for inspiring this code:
    https://github.com/CompVis/taming-transformers
"""
import copy
import math
import os
from typing import Text, Mapping, Tuple, Union, Optional
import torch

from einops import rearrange

from modeling.modules import BaseModel
from modeling.taming.taming_autoencoder import Encoder, Decoder
from modeling.quantizer import SimpleVectorizer as VectorQuantizer


class OriginalVQModel(BaseModel):
    """Legacy model only used for inference/debugging."""
    def __init__(self,
                 config,
                 ):
        super().__init__()

        legacy_conifg = {
            "double_z": False,
            "z_channels": 256,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,1,2,2,4),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0
        }

        self.encoder = Encoder(**legacy_conifg)
        self.decoder = Decoder(**legacy_conifg)
        self.quantize = VectorQuantizer(1024, 256, commitment_cost=0.25)
        self.quant_conv = torch.nn.Conv2d(legacy_conifg["z_channels"], 256, 1)
        self.post_quant_conv = torch.nn.Conv2d(256, legacy_conifg["z_channels"], 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        x = x * 2.0 - 1.0
        z = self.encoder(x)
        z = self.quant_conv(z)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> torch.Tensor:
        z_quantized = self.post_quant_conv(z_quantized)
        decoded = self.decoder(z_quantized)
        decoded = (decoded + 1.0) / 2.0
        return decoded

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z_quantized = self.quantize.get_codebook_entry(tokens)
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z_quantized, result_dict = self.encode(input)
        decoded = self.decode(z_quantized)
        return decoded, result_dict
    
    def load_pretrained(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """The modified load routine does online renaming of keys to account for 
        different names in the pretrained checkpoint.

        Args:
            pretrained_model_path  -> Union[str, os.PathLike]: _description_
            strict_loading -> bool: Flag indicating whether to load strictly. Defaults to True.
            torch_dtype -> Optional[torch.dtype]: Optional pytorch type to use, e.g. bfloat. Defaults to None.

        Raises:
            ValueError: If the path is incorrect or the `torch_dtype` is not a correct type.
        """        
        if os.path.isfile(pretrained_model_path):
            model_file = pretrained_model_path
        elif os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.isfile(pretrained_model_path):
                model_file = pretrained_model_path
            else:
                raise ValueError(f"{pretrained_model_path} does not exist")
        else:
            raise ValueError(f"{pretrained_model_path} does not exist")

        checkpoint = torch.load(model_file, map_location="cpu")

        ignore_keys = ("loss.",)

        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        keys = list(checkpoint.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del checkpoint[k]

        new_checkpoint = copy.deepcopy(checkpoint)

        self.load_state_dict(new_checkpoint, strict=strict_loading)

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default
        self.eval()


if __name__ == "__main__":
    from torchinfo import summary
    model = OriginalVQModel(None, None)
    summary(model, input_size=(1,3,256,256), depth=5, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
