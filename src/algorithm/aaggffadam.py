from .fedadam import FedadamOptimizer



class AaggffadamOptimizer(FedadamOptimizer):
    def __init__(self, params, **kwargs):
        super(AaggffadamOptimizer, self).__init__(params=params, **kwargs)
    