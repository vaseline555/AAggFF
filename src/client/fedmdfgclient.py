import torch
import logging

from .fedsgdclient import FedsgdClient
from src import MetricManager

logger = logging.getLogger(__name__)




class FedmdfgClient(FedsgdClient):
    def __init__(self, **kwargs):
        super(FedmdfgClient, self).__init__(**kwargs)

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        coefs = []
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.model.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # accumulate mini-batch gradients
            for param in self.model.parameters():
                if not hasattr(param, 'grad_buffer'):
                    param.grad_buffer = [param.grad.data]
                else:
                    param.grad_buffer.append(param.grad.data)
            coefs.append(len(targets))
            mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            # get common gradient direction
            coefs = torch.tensor(coefs).float().to(self.args.device)
            coefs = coefs / coefs.sum()
            
            for param in self.model.parameters():
                param.grad.zero_()
                param.grad.data = torch.stack(param.grad_buffer, dim=-1).mul(coefs).sum(-1).type(param.grad.dtype)
                del param.grad_buffer
            
            # calculate loss
            mm.aggregate(len(self.training_set), 1)
            if self.args.C == 1: # log for cross-silo setting only
                log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] <CLIENT {str(self.id)}>'
                log_string += f' ( Loss={mm.results[1]["loss"]:.4f} )'
                for m in self.args.eval_metrics:
                    log_string += f' ( {m.title()}={mm.results[1]["metrics"][m]:.4f} )' 
                logger.info(log_string)
        return mm.results
    