from .fedproxclient import FedproxClient



class AaggffproxClient(FedproxClient):
    def __init__(self, **kwargs):
        super(AaggffproxClient, self).__init__(**kwargs)
    