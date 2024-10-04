import os
import torch
import urllib
import logging

import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)



class MQP(torch.utils.data.Dataset):
    def __init__(self, doctor_id, inputs, targets, tokenizer):
        self.identifier = doctor_id
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.MAX_LEN = 200 
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # get input pairs
        inputs1, inputs2 = self.inputs[index].T
        
        # tokenize and connect inputs
        inputs = self.tokenizer.encode_plus(
            inputs1,
            inputs2,
            add_special_tokens=True, 
            padding='max_length', 
            max_length=self.MAX_LEN, 
            truncation=True
        )
        inputs = torch.stack([torch.tensor(inputs['input_ids']).long(), torch.tensor(inputs['attention_mask']).long()])
        targets = torch.tensor(self.targets[index]).long() 
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch MQP dataset
def fetch_mqp(args, root, tokenizer):
    URL = 'https://raw.githubusercontent.com/curai/medical-question-pair-dataset/master/mqp.csv'
    MD5 = '06c0121fe60cefa3cdf4d49a637dcaa6'

    def _download(root):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL)) as response:
                from tqdm import tqdm
                with open(os.path.join(root, URL.split('/')[-1]), 'wb') as fh, tqdm(total=response.length) as pbar:
                    for chunk in iter(lambda: response.read(1024 * 32), b""):
                        # filter out keep-alive new chunks
                        if not chunk: continue
                        fh.write(chunk)
                        pbar.update(len(chunk))
        except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
            if URL[:5] == 'https':
                url = URL.replace('https:', 'http:')
                with urllib.request.urlopen(urllib.request.Request(url)) as response:
                    from tqdm import tqdm
                    with open(os.path.join(root, URL.split('/')[-1]), 'wb') as fh, tqdm(total=response.length) as pbar:
                        for chunk in iter(lambda: response.read(1024 * 32), b""):
                            # filter out keep-alive new chunks
                            if not chunk: continue
                            fh.write(chunk)
                            pbar.update(len(chunk))
            else:
                logger.exception(e)
                raise Exception(e)
    
    def _create_clients(args, tokenizer, dataset, doctor_id): 
        # drop identifier column
        dataset.drop(columns=['doctor_id'], inplace=True)

        # get inputs and targets
        inputs = dataset.loc[:, ['question_1', 'question_2']].values
        targets = dataset['label'].values

        # train-test split with stratified manner
        train_indices, test_indices = train_test_split(
            dataset.index.tolist(), 
            test_size=args.test_size, 
            random_state=args.seed, 
            stratify=targets 
        )
        train_inputs, train_targets = inputs[train_indices], targets[train_indices]
        test_inputs, test_targets = inputs[test_indices], targets[test_indices]
        
        return (
            MQP(f'[MQP] CLIENT < {doctor_id:^4d} > (train)', train_inputs, train_targets, tokenizer), 
            MQP(f'[MQP] CLIENT < {doctor_id:^4d} > (test)', test_inputs, test_targets, tokenizer)
        ) 

    logger.info(f'[LOAD] [MQP] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'mqp')):
        os.mkdir(os.path.join(root, 'mqp'))
        _download(root=os.path.join(root, 'mqp'))
        logger.info(f'[LOAD] [MQP] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [MQP] ...raw data already exists!')
    
    logger.info(f'[LOAD] [MQP] Create client datsets!')
    raw = pd.read_csv(os.path.join(root, 'mqp', 'mqp.csv'), header=None, names=['doctor_id', 'question_1', 'question_2', 'label'])
    client_datasets = []
    for doctor_id in sorted(raw['doctor_id'].unique()):
        client_datasets.append(_create_clients(args, tokenizer, raw[raw['doctor_id'] == doctor_id].reset_index(drop=True), doctor_id))
    logger.info('[LOAD] [MQP] ...created client datasets!')

    args.num_classes = 2
    args.K = 11
    return {}, client_datasets, args
