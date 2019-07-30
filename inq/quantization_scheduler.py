import math
from functools import partial

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class INQScheduler(object):
    """Handles the the weight partitioning and group-wise quantization stages
    of the incremental network quantization procedure.

    Args:
        optimizer (Optimizer): Wrapped optimizer (use inq.SGD).
        iterative_steps (list): accumulated portions of quantized weights.
        strategy ("random"|"pruning"): weight partition strategy, either random or pruning-inspired.

    Example:
        >>> optimizer = inq.SGD(...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
        >>> for inq_step in range(3):
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)

    """
    def __init__(self, optimizer, iterative_steps, strategy="pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")
        if strategy not in ["random", "pruning"]:
            raise ValueError("INQ supports \"random\" and \"pruning\" -inspired weight partitioning")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.strategy = strategy
        self.idx = 0

        for group in self.optimizer.param_groups:
            group['ns'] = []
            if group['weight_bits'] is None:
                continue
            for p in group['params']:
                if p.requires_grad is False:
                    group['ns'].append((0, 0))
                    continue
                s = torch.max(torch.abs(p.data)).item()
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1 + 1 - (2**(group['weight_bits']-1))/2)
                group['ns'].append((n_1, n_2))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def quantize(self):
        """Quantize the parameters handled by the optimizer.
        """
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                T = group['Ts'][idx]
                ns = group['ns'][idx]
                device = p.data.device
                quantizer = partial(self.quantize_weight, n_1=ns[0], n_2=ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                p.data = torch.where(T == 0, fully_quantized, p.data)

    def quantize_weight(self, weight, n_1, n_2):
        """Quantize a single weight using the INQ quantization scheme.
        """
        alpha = 0
        beta = 2 ** n_2
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_2, n_1 + 1):
            if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(beta, weight)
            alpha = 2 ** i
            beta = 2 ** (i + 1)
        return quantized_weight

    def step(self):
        """Performs weight partitioning and quantization
        """
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                if self.strategy == "random":
                    if self.idx == 0:
                        probability = self.iterative_steps[0]
                    elif self.idx >= len(self.iterative_steps) - 1:
                        probability = 1
                    else:
                        probability = (self.iterative_steps[self.idx] - self.iterative_steps[self.idx - 1]) / (1 - self.iterative_steps[self.idx - 1])

                    T = group['Ts'][idx]
                    T_rand = torch.rand_like(p.data)
                    zeros = torch.zeros_like(p.data)
                    T = torch.where(T_rand <= probability, zeros, T)
                    group['Ts'][idx] = T
                else:
                    zeros = torch.zeros_like(p.data)
                    ones = torch.ones_like(p.data)
                    quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), 1 - self.iterative_steps[self.idx])
                    T = torch.where(torch.abs(p.data) >= quantile, zeros, ones)
                    group['Ts'][idx] = T
                    print("Step: {}".format(self.iterative_steps[self.idx]))

        self.idx += 1
        self.quantize()


def reset_lr_scheduler(scheduler):
    """Reset the learning rate scheduler.
    INQ requires resetting the learning rate every iteration of the procedure.

    Example:
        >>> optimizer = inq.SGD(...)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
        >>> for inq_step in range(3):
        >>>     reset_lr_scheduler(scheduler)
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         scheduler.step()
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)
    """
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    last_epoch = 0
    scheduler.last_epoch = last_epoch
    scheduler.step(last_epoch)

