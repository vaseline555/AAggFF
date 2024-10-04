import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FedsgdClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedsgdClient, self).__init__(**kwargs)

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.model.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.training_set), 1)
            if self.args.C == 1: # log for cross-silo setting only
                log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] <CLIENT {str(self.id)}>'
                log_string += f' ( Loss={mm.results[1]["loss"]:.4f} )'
                for m in self.args.eval_metrics:
                    log_string += f' ( {m.title()}={mm.results[1]["metrics"][m]:.4f} )' 
                logger.info(log_string)
        return mm.results
    
    def upload(self):
        self.model.to('cpu')
        return self.model.named_parameters()
