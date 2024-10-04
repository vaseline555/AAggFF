from .fedavgclient import FedavgClient



class AaggffadamClient(FedavgClient):
    def __init__(self, **kwargs):
        super(AaggffadamClient, self).__init__(**kwargs)
    