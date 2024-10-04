import os
import torch
import logging
import torchaudio

logger = logging.getLogger(__name__)


LABELS = sorted([
    'on', 'learn', 'tree', 'down', 'forward', 
    'backward', 'happy', 'off', 'nine', 'eight', 
    'left', 'four', 'one', 'visual', 'sheila', 
    'no', 'six', 'dog', 'up', 'five', 
    'marvin', 'cat', 'yes', 'zero', 'house', 
    'bird', 'go', 'seven', 'stop', 'wow', 
    'three', 'follow', 'right', 'bed', 'two'
])

# dataset wrapper module
class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, targets, suffix):
        self.dataset = dataset
        self.suffix = suffix
        self.targets = targets

    def __getitem__(self, index):
        def _label_to_index(word):
            # Return the position of the word in labels
            return torch.tensor(LABELS.index(word))

        def _pad_sequence(batch, max_len=16000):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [torch.nn.functional.pad(item.t(), (0, 0, 0, max_len - len(item.t())), value=0.) for item in batch]
            batch = torch.cat(batch)
            return batch.t()

        # get raw batch by index
        batch = self.dataset[index]

        # gather in lists, and encode labels as indices
        inputs, targets = [], []
        for waveform, _, label, *_ in (batch, ):
            inputs += [waveform]
            targets += [_label_to_index(label)]

        # group the list of tensors into a batched tensor
        inputs = _pad_sequence(inputs)
        targets = torch.stack(targets).squeeze()
        return inputs, targets

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[SPEECHCOMMANDS] {self.suffix}'

class SpeechCommands(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, split, download):
        self.data_root = os.path.expanduser(root)
        self.split = split
        self.download = download
        super(SpeechCommands, self).__init__(root=self.data_root, subset=self.split, download=self.download)

    def set_targets(self, targets):
        self.targets = targets

# helper method to fetch CINIC-10 dataset
def fetch_speechcommands(args, root):
    logger.info('[LOAD] [SpeechCommands] Load dataset...!')
    
    # default arguments
    DEFAULT_ARGS = {'root': root, 'download': True, 'split': 'training'}
    
    # create dataset instance
    raw_dataset = SpeechCommands(**DEFAULT_ARGS)
    
    ## convert targets into digits
    targets = torch.tensor([LABELS.index(filename.split('/')[3]) for filename in raw_dataset._walker]).long()
    raw_dataset.set_targets(targets)

    ## assign user identifier
    user_ids = [file_name.split('/')[4].split('_')[0] for file_name in raw_dataset._walker]

    # split by user ID
    logger.info(f'[LOAD] [SpeechCommands] Split by user ID...!')
    client_datasets = []
    for user in list(set(user_ids)):
        indices = [idx for idx, identifier in enumerate(user_ids) if identifier == user]
        subset = torch.utils.data.Subset(raw_dataset, indices)
        if len(subset) <= round(0.5 / args.test_size): # filter out clients having too few samples
            continue
        test_size = round(len(subset) * args.test_size)
        train_size = len(subset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, test_size])

        client_dataset = (
            AudioClassificationDataset(train_dataset, targets[subset.indices][train_dataset.indices], f"< {user.upper()} > (train)"),
            AudioClassificationDataset(test_dataset, targets[subset.indices][test_dataset.indices], f"< {user.upper()} > (test)")
        )
        client_datasets.append(client_dataset)
    logger.info(f'[LOAD] [SpeechCommands] ...done splitting!')

    # adjust arguments
    args.in_channels = args.embedding_size = 1
    args.num_classes = len(torch.unique(torch.as_tensor(targets))) # 35
    args.K = len(client_datasets) # 2005
    return {}, client_datasets, args
