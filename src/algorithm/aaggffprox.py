from .fedprox import FedproxOptimizer



class AaggffproxOptimizer(FedproxOptimizer):
    def __init__(self, params, **kwargs):
        super(AaggffproxOptimizer, self).__init__(params=params, **kwargs)
    