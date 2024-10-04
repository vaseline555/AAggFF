from .fedavgclient import FedavgClient



class TermClient(FedavgClient):
    def __init__(self, **kwargs):
        super(TermClient, self).__init__(**kwargs)