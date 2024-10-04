import os
import torch
import logging
import torchtext
import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)



# split setting adapted from Agnostic Federated Learning (Mohri et al., 2019)
class Adult(torch.utils.data.Dataset):
    def __init__(self, education, inputs, targets):
        self.identifier = education
        self.inputs, self.targets = inputs, targets
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor(self.targets[index]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch Adult dataset
def fetch_adult(args, root):
    URL = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    ]
    MD5 = [
        '5d7c39d7b8804f071cdd1f2a7c460872',
        '35238206dfdf7f1fe215bbb874adecdc'
    ]
    COL_NAME = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',\
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',\
        'house_per_week', 'native_country', 'targets'
    ]
    NUM_COL = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'house_per_week']
    
    def _download(root):
        for idx, (url, md5) in enumerate(zip(URL, MD5)):
            _ = torchtext.utils.download_from_url(
                url=url, 
                root=root, 
                hash_value=md5, 
                hash_type='md5'
            )
            os.rename(os.path.join(root, url.split('/')[-1]), os.path.join(root, f"adult_{'train' if idx == 0 else 'test'}.csv"))
    
    def _munge_and_create_clients(root):
        # load data
        df = pd.read_csv(os.path.join(root, 'adult_train.csv'), header=None, names=COL_NAME, na_values=' ?').reset_index(drop=True)
        df_id = df['education'].str.contains('Doctorate').astype(int)
        df = df.drop(columns=NUM_COL)
        df = pd.concat([pd.get_dummies(df.iloc[:, :-1], columns=[col for col in df.columns if col not in NUM_COL][:-1], drop_first=False, dtype=int), df[['targets']]], axis=1)

        # encode target
        replace_map = {key: value for value, key in enumerate(sorted(df['targets'].unique()))}
        df['targets'] = df['targets'].replace(replace_map)

        # creat clients by education
        clients = {}
        clients['phd'] = df.loc[df_id == 1]
        clients['non-phd'] = df.loc[df_id == 0]
        return clients
    
    def _process_client_datasets(identifier, dataset, seed, test_size):
        # get inputs and targets
        inputs, targets = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size, random_state=seed, stratify=targets)
        return (
            Adult(f'[ADULT] CLIENT < {identifier.upper()} > (train)', train_inputs, train_targets), 
            Adult(f'[ADULT] CLIENT < {identifier.upper()} > (test)', test_inputs, test_targets)
        ) 
        
    logger.info(f'[LOAD] [ADULT] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'adult')):
        _download(root=os.path.join(root, 'adult'))
        logger.info(f'[LOAD] [ADULT] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [ADULT] ...raw data already exists!')
    
    logger.info(f'[LOAD] [ADULT] Munging dataset and create clients!')
    raw_clients = _munge_and_create_clients(os.path.join(root, 'adult'))
    logger.info('[LOAD] [ADULT] ...munged dataset and created clients!!')
    
    logger.info(f'[LOAD] [ADULT] Processing client datsets!')
    client_datasets = []
    for identifier, dataset in raw_clients.items():
        client_datasets.append(_process_client_datasets(identifier, dataset, args.seed, args.test_size))
    logger.info('[LOAD] [ADULT] ...processed client datasets!')

    args.in_features = 99
    args.num_classes = 2
    args.K = 2
    return {}, client_datasets, args
