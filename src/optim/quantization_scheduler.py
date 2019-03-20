import math
from functools import partial

import torch
from torch.optim.optimizer import Optimizer


class INQScheduler(object):
    def __init__(self, optimizer, iterative_steps, weight_bits=3):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.idx = 0
        self.step()

        for group in self.optimizer.param_groups:
            group['ns'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['ns'].append((0, 0))
                    continue
                s = torch.max(torch.abs(p.data)).item()
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1 + 1 - (2**(weight_bits-1))/2)
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
        # quantize the parameters
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                ns = group['ns'][idx]
                T = group['Ts'][idx]
                device = p.data.device
                quantizer = partial(self.quantize_weight, n_1=ns[0], n_2=ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                p.data = torch.where(T == 0, fully_quantized, p.data)

    def quantize_weight(self, weight, n_1, n_2):
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
        # update T matrix
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue

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

        self.idx += 1
