"""This file contains code to run different learning rate schedulers.

We thank the following public implementations for inspiring this code:
    https://github.com/huggingface/open-muse/blob/main/muse/lr_schedulers.py
"""
import math
from enum import Enum
from typing import Optional, Union

import torch


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_MINIMUM = "cosine_with_minimum"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


def get_constant_schedule(
    optimizer: torch.optim.Optimizer,
    last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which
    the learning rate increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr
    set in the optimizer to 0, after a warmup period during which it increases linearly
    from 0 to the initial lr set in the optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        num_training_steps -> int:
            The total number of training steps.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /  \
                float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_minimum_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    minimum_rate: float,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the
    cosine function between the initial lr set in the optimizer to the minumum rate,
    after a warmup period during which it increases linearly between 0 and the initial
    lr set in the optimizer.

    Inspired by https://arxiv.org/pdf/1608.03983.pdf. 

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        num_training_steps -> int:
            The total number of training steps.
        minimum_rate -> float:
            The minimum rate of the base lr to anneal to.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        
        cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, cos_term + minimum_rate - minimum_rate * cos_term)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the
    cosine function between the initial lr set in the optimizer to 0, after a warmup period
    during which it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        num_training_steps -> int:
            The total number of training steps.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the 
    cosine function between the initial lr set in the optimizer to 0, with several hard
    restarts, after a warmup period during which it increases linearly between 0 and the 
    initial lr set in the optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        num_training_steps -> int:
            The total number of training steps.
        num_cycles -> int:
            The number of hard restarts to use. Defaults to 1.
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: int = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the
    initial lr set in the optimizer to end lr defined by *lr_end*, after a warmup period
    during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer -> torch.optim.Optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps -> int:
            The number of steps for the warmup phase.
        num_training_steps -> int:
            The total number of training steps.
        lr_end -> int:
            The end LR. Defaults to 1e-7
        power -> float:
            Power factor. Default to 1.0
        last_epoch -> int:
            The index of the last epoch when resuming training. Defaults to -1.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_MINIMUM: get_cosine_schedule_with_minimum_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    minimum_rate: float = 0.1,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name -> Union[str, `SchedulerType`]:
            The name of the scheduler to use.
        optimizer -> torch.optim.Optimizer:
            The optimizer that will be used during training.
        num_warmup_steps -> Optional[int]:
            The number of warmup steps to do. This is not required by all schedulers (hence 
            the argument being optional), the function will raise an error if it's unset and
            the scheduler type requires it.
        num_training_steps -> Optional[int]:
            The number of training steps to do. This is not required by all schedulers (hence
            the argument being optional), the function will raise an error if it's unset and
            the scheduler type requires it.
        num_cycles -> int:
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler. Defaults to 1.
        power -> float:
            Power factor. See `POLYNOMIAL` scheduler. Defaults to 1.0
        minimum_rate -> float:
            The minimum rate of the base lr to anneal to. Defaults to 0.1
        
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )
    if name == SchedulerType.COSINE_WITH_MINIMUM:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            minimum_rate=minimum_rate,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )