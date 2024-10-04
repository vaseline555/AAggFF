from .fedavgclient import FedavgClient



class FedmgdaClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedmgdaClient, self).__init__(**kwargs)