import torch
from torch.optim.optimizer import Optimizer


class INQScheduler(object):
    def __init__(self, optimizer, iterative_steps):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.idx = 0
        self.step()

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
        raise NotImplemented

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
