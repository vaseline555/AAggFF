from .fedavg import FedavgOptimizer



class AflOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(AflOptimizer, self).__init__(params=params, **kwargs)
