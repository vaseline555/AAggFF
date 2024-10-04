from .fedavg import FedavgOptimizer



class TermOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(TermOptimizer, self).__init__(params=params, **kwargs)