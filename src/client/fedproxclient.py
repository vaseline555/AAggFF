import copy
import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FedproxClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedproxClient, self).__init__(**kwargs)

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): 
            param.requires_grad = False

        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )
        
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                prox = 0.
                for name, param in self.model.named_parameters():
                    prox += (param - global_model.get_parameter(name)).norm(2)
                loss = loss.add(prox.mul(0.5).mul(self.args.mu))

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()
                
                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
                if self.args.C == 1: # log for cross-silo setting only
                    log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] <CLIENT {str(self.id)}> [Local Epoch|{e + 1:3d}/{self.args.E:3d}]'
                    log_string += f' ( Loss={mm.results[e + 1]["loss"]:.4f} )'
                    for m in self.args.eval_metrics:
                        log_string += f' ( {m.title()}={mm.results[e + 1]["metrics"][m]:.4f} )' 
                    logger.info(log_string)
        else:
            self.model.to('cpu')
        return mm.results
    