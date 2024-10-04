import os
import torch
import logging
import torchvision

logger = logging.getLogger(__name__)



# split setting adapted from Agnostic Federated Learning (Mohri et al., 2019)
class FFMNIST(torch.utils.data.Dataset): # Fair Fashion-MNIST
    def __init__(self, dataset, assigned_class, suffix):
        self.dataset = dataset
        self.suffix = suffix
        self.targets = torch.ones(len(dataset)).mul(assigned_class).long()

    def __getitem__(self, index):
        inputs = self.dataset[index][0]
        targets = self.targets[index]
        return inputs, targets

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[FMNIST] {self.suffix}'

# helper method to fetch Fashion MNIST dataset
def fetch_ffmnist(args, root, transform):
    def _create_clients(dataset):
        class_indices = [dataset.classes.index('Shirt'), dataset.classes.index('Pullover'), dataset.classes.index('T-shirt/top')]
        clients = {}
        for class_index in class_indices:
            clients[dataset.classes[class_index]] = torch.utils.data.dataset.Subset(dataset, torch.where(dataset.targets == class_index)[0])
        return clients
    
    def _process_client_datasets(idx, name, dataset, seed, test_size):
        g = torch.Generator().manual_seed(seed)
        num_test = int(len(dataset) * test_size)
        num_train = len(dataset) - num_test
        train_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_test], generator=g)
        return (
            FFMNIST(train_set, idx, f'CLIENT < {name} > (train)'), 
            FFMNIST(test_set, idx, f'CLIENT < {name} > (test)')
        ) 
        
    logger.info(f'[LOAD] [FMNIST] Check if raw data exists; if not, start downloading!')
    raw_dataset = torchvision.datasets.__dict__['FashionMNIST'](root=os.path.join(root, 'fmnist'), download=True, train=True, transform=transform)
    
    logger.info(f'[LOAD] [FMNIST] Create clients!')
    raw_clients = _create_clients(raw_dataset)
    logger.info('[LOAD] [FMNIST] ...created clients!!')
    
    logger.info(f'[LOAD] [FMNIST] Process client datsets!')
    client_datasets = []
    for idx, (name, dataset) in enumerate(raw_clients.items()):
        client_datasets.append(_process_client_datasets(idx, name, dataset, args.seed, args.test_size))
    logger.info('[LOAD] [FMNIST] ...processed client datasets!')
    
    args.in_features = 784
    args.num_classes = 3
    args.K = 3
    return {}, client_datasets, args
