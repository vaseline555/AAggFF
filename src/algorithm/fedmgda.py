import torch

from .basealgorithm import BaseOptimizer



class FedmgdaOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        defaults = dict(lr=lr)
        BaseOptimizer.__init__(self); torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)
        self.scale = []

    def step(self, mix_coefs, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups: 
            for param in group['params']:
                if not hasattr(param, 'delta_buffer'): 
                    continue
                deltas = torch.stack(param.delta_buffer)
                delta = deltas.transpose(0, -1).data * mix_coefs * torch.tensor(self.scale).reciprocal().float()
                delta = delta.sum(-1, keepdim=True).transpose(-1, 0).squeeze(0)
                param.data.sub_(delta.mul(group['lr']))
                del param.delta_buffer 
        else:
            self.scale = []

    def accumulate(self, local_layers_iterator, check_if=lambda name: 'some_modules' in name):
        flattened_delta = []
        for idx, group in enumerate(self.param_groups):
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if 'num_batches_tracked' in name:
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                if ((idx == 1) and server_param.dtype == torch.long):
                    continue
                if check_if(name):
                    pass

                ## for collecting \Delta w_k^t
                local_delta = (server_param - local_signals).data.type(server_param.dtype)
                if not hasattr(server_param, 'delta_buffer'): 
                    server_param.delta_buffer = [local_delta]
                else:
                    server_param.delta_buffer.append(local_delta)

                ## track the norm of g_i
                flattened_delta.append(local_delta.data.view(-1))
        delta_norm = torch.cat(flattened_delta).data.norm(2)
        self.scale.append(delta_norm)                
        return torch.cat(flattened_delta).div(delta_norm)
