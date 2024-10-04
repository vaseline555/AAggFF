from .fedavg import FedavgOptimizer



class PropfairOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(PropfairOptimizer, self).__init__(params=params, **kwargs)
