import copy
import torch
import inspect
import logging
import itertools

from .baseclient import BaseClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self.optim = torch.optim.__dict__[self.args.optimizer]
        criterion = torch.nn.__dict__[self.args.criterion]
        if self.args.use_class_weights:
            self.class_weights = torch.ones(self.args.num_classes if self.args.num_classes > 1 else 2).to(self.args.device)
            for i in range(len(self.training_set)):
                c = self.training_set.targets[i]
                self.class_weights[c] += 1
            self.class_weights = self.class_weights.div(len(self.training_set) + self.args.num_classes)
            self.class_weights = self.class_weights.reciprocal()
            self.criterion = criterion(weight=self.class_weights)
        else:
            self.criterion = criterion()

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle, pin_memory=self.args.C == 1)
    
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
                
                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()

                mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
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

    @torch.no_grad()
    def evaluate(self, need_feedback=False):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.train_loader if need_feedback else self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.training_set) if need_feedback else len(self.test_set))
        return mm.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        self.model.to('cpu')
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])
    
    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
