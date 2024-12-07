"""This file contains the definition of base classes.

We thank the following public implementations for inspiring this code:
    https://github.com/huggingface/open-muse/blob/main/muse/modeling_utils.py
"""

import os
from typing import Union, Callable, Tuple, Dict, Optional, List

import torch


def get_parameter_device(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """ Save a model to a directory, so that it can be re-loaded using the
        load_pretrained class method.

        Args:
            save_directory -> Union[str, os.PathLike]: Directory to which to save. Will be created
                if it doesn't exist.
            save_function -> Optional[Callable]: The function to use to save the state dictionary.
                Useful on distributed training like TPUs when one need to replace `torch.save` by another method.
            state_dict -> Optional[Dict[str, torch.Tensor]]: The state dictionary to save. If `None`, the model's
                state dictionary will be saved.
        """
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        weights_name = "pytorch_model.bin"

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def load_pretrained(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """ Instantiate a pretrained pytorch model from a weights path.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you should first set it back in training mode with `model.train()`.

        Args:
            pretrained_model_path -> Union[str, os.PathLike]: Path to a pretrained model.
            strict_loading -> bool: Whether or not to strictly enforce that the provided weights file matches the
                architecture of this model.
            torch_dtype -> Optional[torch.dtype]: The dtype to use for the model. Defaults to `None`, which means
                no conversion.
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

        self.load_state_dict(checkpoint, strict=strict_loading)

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default
        self.eval()


    @property
    def device(self):
        """ Returns the device of the model.

        Returns:
            `torch.device`: The device of the model.
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """ Returns the dtype of the model."""
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

