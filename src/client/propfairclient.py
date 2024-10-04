import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class PropfairClient(FedavgClient):
    def __init__(self, **kwargs):
        super(PropfairClient, self).__init__(**kwargs)

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )
        
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # induce proportional fairness
                eps = 0.2
                if self.args.fair_const - loss < eps:           
                    propfair_loss = loss.div(self.args.fair_const)
                else:
                    propfair_loss = loss.div(self.args.fair_const).mul(-1).add(1.).log().mul(-1)

                optimizer.zero_grad()
                propfair_loss.backward()
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
    