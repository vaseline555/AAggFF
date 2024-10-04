from .fedyogi import FedyogiOptimizer



class AaggffyogiOptimizer(FedyogiOptimizer):
    def __init__(self, params, **kwargs):
        super(AaggffyogiOptimizer, self).__init__(params=params, **kwargs)
    