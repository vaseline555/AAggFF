from .fedavgclient import FedavgClient



class AflClient(FedavgClient):
    def __init__(self, **kwargs):
        super(AflClient, self).__init__(**kwargs)