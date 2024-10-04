
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class PropfairServer(FedavgServer):
    def __init__(self, **kwargs):
        super(PropfairServer, self).__init__(**kwargs)
