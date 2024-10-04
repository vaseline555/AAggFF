from .fedavgclient import FedavgClient



class QfedavgClient(FedavgClient):
    def __init__(self, **kwargs):
        super(QfedavgClient, self).__init__(**kwargs)