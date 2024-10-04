from .fedadagrad import FedadagradOptimizer



class AaggffadagradOptimizer(FedadagradOptimizer):
    def __init__(self, params, **kwargs):
        super(AaggffadagradOptimizer, self).__init__(params=params, **kwargs)
    