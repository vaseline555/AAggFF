import os
import torch
import logging
import torchvision

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class Berka(torch.utils.data.Dataset):
    def __init__(self, region, inputs, targets, scaler):
        self.identifier = region
        self.inputs, self.targets = inputs, targets
        self.scaler = scaler
    
    @staticmethod
    def inverse_transform(self, inputs):
        assert inputs.shape[-1] == 13
        return self.scaler.inverse_transform(inputs[:, :-2])
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor(self.targets[index]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch Berka dataset
def fetch_berka(args, root):
    URL = 'http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip'
    MD5 = '88d001738478c5e348a0896878f2433c'

    def _download(root):
        torchvision.datasets.utils.download_and_extract_archive(
            URL, root, filename='data_berka.zip',
            remove_finished=True, md5=MD5
        )
    
    def _munge_and_split(root):
        dfs = dict()
        for file in os.listdir(root):
            dfs[file.split('.')[0]] = pd.read_csv(
                os.path.join(root, file), 
                delimiter=';', 
                low_memory=False
            )

        # merge loan and account dataframes
        loan_account = pd.merge(dfs['loan'], dfs['account'], on ='account_id', suffixes=['_loan', '_acc'])
        loan_account['status'] = loan_account['status'].replace({'A': 0, 'C': 0, 'B': 1, 'D': 1})
        loan_account['date_between'] = pd.to_datetime(loan_account['date_loan'], format='%y%m%d').sub(pd.to_datetime(loan_account['date_acc'], format='%y%m%d')).dt.days
        loan_account.drop(columns=['date_loan', 'date_acc'], inplace=True)

        # merge district dataframe
        district = dfs['district']
        district = district.rename(columns={'A1': 'district_id'})
        merged = pd.merge(
            loan_account, 
            district[['district_id', 'A3', 'A4', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']], 
            on='district_id', 
            how='left'
        )
        merged = merged.rename(columns={'A4': 'num_inhabitants', 'A11': 'average_salary', 'A14': 'entrepreneur_rate'})

        merged['average_unemployment_rate'] = merged[['A12', 'A13']].mean(axis=1, numeric_only=True)
        merged.drop(columns=['A12', 'A13'], inplace=True)

        merged['average_crime_rate'] = merged[['A15', 'A16']].mean(axis=1, numeric_only=True)
        merged.drop(columns=['A15', 'A16'], inplace=True)

        # merge order dataframe
        order = dfs['order']
        order = order.rename(columns={'amount': 'order_amount'})
        order = order.loc[order['account_id'].isin(merged['account_id'].unique()), ['account_id', 'order_amount']]
        merged = pd.merge(merged, order.groupby('account_id').mean().reset_index(), on='account_id', how='left')
        merged = merged.rename(columns={'order_amount': 'average_order_amount'})

        # merge transaction dataframe
        trans = dfs['trans']
        trans = trans[['account_id', 'amount', 'balance']]
        trans = trans.rename(columns={'amount': 'trans_amount', 'balance': 'trans_balance'})
        
        num_trans = trans.groupby('account_id').count().reset_index()[['account_id', 'trans_amount']].rename(columns={'trans_amount': 'num_trans'})
        merged = merged.merge(num_trans, on='account_id')

        avg_trans = trans.groupby('account_id').mean().reset_index().rename(columns={'trans_amount': 'average_trans_amount', 'trans_balance': 'average_trans_balance'})
        merged = merged.merge(avg_trans, on='account_id')
        merged = merged.drop(columns=['loan_id', 'account_id', 'district_id'])
        merged['frequency'] = merged['frequency'].replace({'POPLATEK MESICNE': 0, 'POPLATEK TYDNE': 1, 'POPLATEK PO OBRATU': 2})
        return merged

    def _create_clients(args, dataset, region): 
        df = dataset[dataset['A3'] == region].copy() 
        df.drop(columns=['A3'], inplace=True)

        # get one-hot encoded dummy columns for categorical data ('frequency')
        df = pd.get_dummies(df, columns=['frequency'], drop_first=True, dtype=int)
        
        # get inputs and targets
        inputs, targets = df.drop(columns=['status']).values, df['status'].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            inputs, targets, 
            test_size=args.test_size, 
            random_state=args.seed, 
            stratify=targets 
        )
        
        # scaling inputs
        scaler = StandardScaler()
        train_inputs[:, :-2] = scaler.fit_transform(train_inputs[:, :-2])
        test_inputs[:, :-2] = scaler.transform(test_inputs[:, :-2])

        return (
            Berka(f'[BERKA] CLIENT < {region} > (train)', train_inputs, train_targets, scaler), 
            Berka(f'[BERKA] CLIENT < {region} > (test)', test_inputs, test_targets, scaler)
        ) 

    logger.info(f'[LOAD] [BERKA] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'berka')):
        _download(root=os.path.join(root, 'berka'))
        logger.info(f'[LOAD] [BERKA] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [BERKA] ...raw data already exists!')
    
    logger.info(f'[LOAD] [BERKA] Check file extension...!')
    for file in os.listdir(os.path.join(root, 'berka')):
        if file.endswith('.asc'):
            os.rename(os.path.join(root, 'berka', file), os.path.join(root, 'berka', f'{file.split(".")[0]}.csv'))
    logger.info(f'[LOAD] [BERKA] ...checked file extension (.csv)!')

    logger.info(f'[LOAD] [BERKA] Munging and splitting dataset!')
    proc_dataset = _munge_and_split(os.path.join(root, 'berka'))
    logger.info('[LOAD] [BERKA] ...munged and splitted dataset!')

    logger.info(f'[LOAD] [BERKA] Create client datsets!')
    client_datasets = []
    for region in proc_dataset['A3'].value_counts().index:
        if region == 'north Bohemia': continue # 'north Bohemia' has only one default...
        client_datasets.append(_create_clients(args, proc_dataset, region))
    logger.info('[LOAD] [BERKA] ...created client datasets!')

    args.in_features = 15
    args.num_classes = 2
    args.K = 7
    return {}, client_datasets, args
