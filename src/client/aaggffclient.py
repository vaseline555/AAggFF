from .fedavgclient import FedavgClient



class AaggffClient(FedavgClient):
    def __init__(self, **kwargs):
        super(AaggffClient, self).__init__(**kwargs)