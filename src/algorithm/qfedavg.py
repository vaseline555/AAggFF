import torch

from .basealgorithm import BaseOptimizer



class QfedavgOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lipschitz = kwargs.get('lipschitz')
        q = kwargs.get('q')
        lr = kwargs.get('lr')
        defaults = dict(lipschitz=lipschitz, q=q, lr=lr)
        BaseOptimizer.__init__(self); torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)
        self.numerator, self.denominator = [], []

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups: 
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data.div(sum(self.denominator))
                param.data.sub_(delta.mul(group['lr']))
        else:
            mix_coefs = torch.tensor(self.numerator).div(sum(self.denominator)).float()
            self.numerator, self.denominator = [], []
        return mix_coefs

    def accumulate(self, local_layers_iterator, loss, check_if=lambda name: 'some_modules' in name):
        flattened_delta = []
        for idx, group in enumerate(self.param_groups):
            lipschitz, q = group['lipschitz'], group['q']
            loss_tensor = torch.tensor(loss)
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if 'num_batches_tracked' in name:
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                if ((idx == 1) and server_param.dtype == torch.long):
                    continue
                if check_if(name):
                    pass

                ## for collecting (w^t - \bar{w}_k^{t+1}; same as FedAvg)
                local_delta = (server_param - local_signals).data.type(server_param.dtype).mul(lipschitz)
                if server_param.grad is None: # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta.mul(torch.float_power(loss_tensor, q))
                else:
                    server_param.grad.data.add_(local_delta.mul(torch.float_power(loss_tensor, q)))

                ## for the squared norm of (w^t - \bar{w}_k^{t+1})
                flattened_delta.append(local_delta.data.view(-1))
        delta_norm = torch.cat(flattened_delta).pow(2).sum()

        ## for collecting mixing coefficient
        local_h = delta_norm.mul(torch.float_power(loss_tensor, (q - 1)).mul(q)).add(torch.float_power(loss_tensor, q).mul(lipschitz))
        self.numerator.append(torch.float_power(loss_tensor, q).mul(lipschitz))
        self.denominator.append(local_h)
