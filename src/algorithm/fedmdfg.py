import math
import torch

from .fedsgd import FedsgdOptimizer



class FedmdfgOptimizer(FedsgdOptimizer):
    def __init__(self, params, **kwargs):
        super(FedmdfgOptimizer, self).__init__(params=params, **kwargs)

    def step(self, grads, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups: 
            for param in group['params']:
                if not param.requires_grad:
                    continue
                flattened_shape = math.prod(param.data.shape)
                grad = grads[:flattened_shape].view(*param.data.shape)
                param.data.sub_(grad.mul(group['lr']))
                grads = grads[flattened_shape:]
