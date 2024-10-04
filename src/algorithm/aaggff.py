from .fedavg import FedavgOptimizer



class AaggffOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(AaggffOptimizer, self).__init__(params=params, **kwargs)