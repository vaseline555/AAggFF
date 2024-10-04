import os
import logging
import importlib

from src.datasets.flamby import *

logger = logging.getLogger(__name__)

    

URL = {
    'Heart': [
        'https://archive.ics.uci.edu/dataset/45/heart+disease',
        {
            'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
            'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
            'va': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
        }
    ],
    'ISIC2019': [
        'https://challenge.isic-archive.com/data/',
        {
            'inputs': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip',
            'metadata': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv',
            'gt': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv',
            'HAM10000_metadata': 'https://github.com/owkin/FLamby/raw/main/flamby/datasets/fed_isic2019/HAM10000_metadata'
        }
    ],
    'IXITiny': [
        'https://brain-development.org/ixi-dataset/',
        {
            'raw': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip'
        }
    ]
 
}
MD5 = { # md5 checksum if direct URL link is provided, file name if Google Drive link ID is provided  
    'Heart': {
        'cleveland': '2d91a8ff69cfd9616aa47b59d6f843db',
        'hungarian': '22e96bee155b5973568101c93b3705f6',
        'switzerland': '9a87f7577310b3917730d06ba9349e20',
        'va': '4249d03ca7711e84f4444768c9426170'
    },
    'ISIC2019': {
        'inputs': '0ecdc9554ef6273b04e59a0bc420ca9d',
        'metadata': '8ca5fdef200ffd8579f48c823b504f7e',
        'gt': '2c02bdcc6e7f36d355f4f86b210595ae',
        'HAM10000_metadata': 'eeb8a69c3654ac8603574a6c3670397f' 
    },
    'IXITiny': {
        'raw': 'eecb83422a2685937a955251fa45cb03'
    }
}

def fetch_flamby(args, dataset_name, root):
    # accept license
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] Need to accept license to use this dataset...!')
    accept_license(URL[dataset_name][0], dataset_name)
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] ...you have accepted license to use the dataset!')

    # download data
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(f'{root}/{dataset_name.lower()}'):
        os.makedirs(os.path.join(root, f'{dataset_name.lower()}'))
        download_data(download_root=f'{root}/{dataset_name.lower()}', dataset_name=dataset_name.lower(), url_dict=URL[dataset_name][-1], md5_dict=MD5[dataset_name])
        logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] ...raw data already exists!')
    
    # fetch data
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] Fetch data...!')
    _, client_datasets, args = importlib.import_module(f'.flamby.{dataset_name.lower()}', package=__package__).fetch_dataset(args, root)
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] ...done fetching data!')
    return {}, client_datasets, args
