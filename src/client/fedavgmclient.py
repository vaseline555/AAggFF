from .fedavgclient import FedavgClient



class FedavgmClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedavgmClient, self).__init__(**kwargs)