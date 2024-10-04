from .fedavgclient import FedavgClient



class AaggffyogiClient(FedavgClient):
    def __init__(self, **kwargs):
        super(AaggffyogiClient, self).__init__(**kwargs)
    