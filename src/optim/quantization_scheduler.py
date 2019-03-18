import torch
from torch.optim.optimizer import Optimizer


class INQScheduler(object):
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        #
        for group in optimizer.param_groups:
            group['Ts'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue

                T = torch.zeros_like(p.data)
                T_size = T.size()
                T_flattened = torch.reshape(T, (-1,))
                T_flattened_size = T_flattened.shape[0]
                T_flattened[:T_flattened_size // 2] = 1
                T = torch.reshape(T_flattened, T_size)
                group['Ts'].append(T)


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
        raise NotImplemented
