import os
import io
import torch
import shutil
import logging
import torchvision
import concurrent.futures

import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from src import TqdmToLogger

logger = logging.getLogger(__name__)



class ISIC2019(torch.utils.data.Dataset):
    def __init__(self, hospital_name, base_path, img_path, targets, transform=None):
        self.name = hospital_name
        self.base_path = base_path
        self.img_path = img_path
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.base_path, f'{self.img_path[idx]}.jpg'))
        label = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label).long()

    def __len__(self):
        return len(self.img_path)

    def __repr__(self):
        return self.name

# helper method to fetch ISIC2019 Melanoma classification dataset
def fetch_dataset(args, root):
    HOSPITAL_NAME = ['BCN', 'HAM_vidir_molemax', 'HAM_vidir_modern', 'HAM_rosendahl', 'MSK', 'HAM_vienna_dias']
    RESIZE = 224
    TRAIN_TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomAffine(degrees=0, shear=(-0.1, 0.1)),
        torchvision.transforms.RandomCrop((200, 200)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),           
    ])
    TEST_TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((200, 200)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def _resize_and_maintain(raw_path, resized_path, size):
        if not str(raw_path).endswith('.jpg'):
            return
        img = Image.open(raw_path)
        old_size = img.size
        ratio = float(size) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, resample=Image.BILINEAR)
        img.save(resized_path.replace('_downsampled', ''))

    def _get_client_datasets(args, hospital, inputs_path, metadata, train_transform, test_transform):
        raw_info = metadata[metadata['dataset'] == hospital]
        raw_inputs, raw_targets = raw_info['image'].values.tolist(), raw_info['targets'].values.tolist()
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            raw_inputs, raw_targets, 
            test_size=args.test_size, 
            random_state=args.seed, 
            stratify=raw_targets
        )
        return (
            ISIC2019(f'[ISIC2019] CLIENT < {hospital} > (train)', inputs_path, train_inputs, train_targets, train_transform),
            ISIC2019(f'[ISIC2019] CLIENT < {hospital} > (test)', inputs_path, test_inputs, test_targets, test_transform)
        )

    # define base path
    base_path = os.path.join(root, 'isic2019')

    # check HAM10000 metadata
    logger.info('[LOAD] [FLAMBY - ISIC2019] Check if HAM10000 metadata is processed...!')
    if os.path.exists(os.path.join(base_path, 'HAM10000_metadata')):
        logger.info('[LOAD] [FLAMBY - ISIC2019] Create HAM10000 metadata...!')
        
        # load BufferedReader type raw metadata
        with open(os.path.join(base_path, 'HAM10000_metadata'), 'rb') as file:
            buffer  = io.FileIO(file.fileno())
            loaded = io.BufferedReader(buffer)
            raw_metadata = loaded.read()
            raw_metadata = str(raw_metadata, 'utf-8')
        
        # parse into readable type
        proc_metadata = raw_metadata.split('\n')
        proc_metadata = [entry.split(',') for entry in proc_metadata if len(entry.split(',')) == len(proc_metadata[0].split(','))]
        
        # convert into csv
        metadata = pd.DataFrame(proc_metadata)

        # save csv and remove original raw file
        os.remove(os.path.join(base_path, 'HAM10000_metadata'))
        metadata.to_csv(os.path.join(base_path, 'HAM10000_metadata.csv'), index=False, header=False)
    elif os.path.exists(os.path.join(base_path, 'HAM10000_metadata.csv')):
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...checked the HAM10000 metadata!')
    else:
        err = '[LOAD] [FLAMBY - ISIC2019] It seems the HAM10000 metadata file is crashed... please download it again!'
        logger.exception(err)
        raise Exception(err)

    # munge metadata
    logger.info('[LOAD] [FLAMBY - ISIC2019] Check the existence of aggreagted metadata...!')
    if not os.path.exists(os.path.join(base_path, 'agg_metadata.csv')):
        logger.info('[LOAD] [FLAMBY - ISIC2019] Start aggreagting all metadata...!')
        
        ## read all metadata
        isic_metadata = pd.read_csv(os.path.join(base_path, 'ISIC_2019_Training_Metadata.csv'))
        isic_gt = pd.read_csv(os.path.join(base_path, 'ISIC_2019_Training_GroundTruth.csv'))
        ham_metadata = pd.read_csv(os.path.join(base_path, 'HAM10000_metadata.csv'))

        ## process HAM10000
        ham_metadata.rename(columns={'image_id': 'image'}, inplace=True)
        ham_metadata.drop(['age', 'sex', 'localization', 'lesion_id', 'dx', 'dx_type'], axis=1, inplace=True)
        
        ## remove images where hospital is not available
        for i, row in isic_metadata.iterrows():
            try:
                if pd.isnull(row['lesion_id']):
                    image = row['image']
                    os.remove(os.path.join(root, 'isic2019', 'ISIC_2019_Training_Input', f'{image}.jpg'))
                    isic_gt = isic_gt.drop(i)
                    isic_metadata = isic_metadata.drop(i)
            except:
                continue
        else:
            isic_metadata['dataset'] = isic_metadata['lesion_id'].str[:4]

        ## join with HAM10000 metadata
        joined = pd.merge(isic_metadata, ham_metadata, how='left', on='image')
        joined['dataset'] = joined['dataset_x'] + joined['dataset_y'].fillna('')
        joined.drop(['dataset_x', 'dataset_y', 'lesion_id'], axis=1, inplace=True)
        joined['dataset'].replace({'BCN_': 'BCN', 'MSK4': 'MSK'}, inplace=True)
        joined.dropna(inplace=True)
        joined.reset_index(drop=True, inplace=True)
        joined['image'] = joined['image'].str.replace('_downsampled', '')

        ## join with target data
        isic_gt['image'] = isic_gt['image'].str.replace('_downsampled', '')
        aggregated = pd.merge(isic_gt, joined, how='inner', on='image')
        targets = pd.DataFrame(aggregated.iloc[:, 1:9].values.astype(int).argmax(1), columns=['targets'])
        aggregated = pd.concat([aggregated['image'], targets, aggregated['dataset']], axis=1)

        ## save
        aggregated.to_csv(os.path.join(base_path, 'agg_metadata.csv'), index=False)
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...successfullly aggreagted all metadata!')
    else:
        aggregated = pd.read_csv(os.path.join(base_path, 'agg_metadata.csv'))
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...aggreagted metadata is already existed!')
    aggregated = aggregated[aggregated['image'] != 'ISIC_0029564'].reset_index(drop=True)

    # preprocessing: resize input images to be 224 x 224
    logger.info('[LOAD] [FLAMBY - ISIC2019] Check if resizing is completed...!')
    if not os.path.exists(os.path.join(base_path, 'resized')):
        logger.info('[LOAD] [FLAMBY - ISIC2019] Resize raw images...(it will take a few minutes)!')

        ## make directory to store resized images
        os.makedirs(os.path.join(base_path, 'resized'))
        samples_path = os.path.join(root, 'isic2019', 'ISIC_2019_Training_Input')

        ## parallel processing
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as workhorse:
            for sample_path in TqdmToLogger(
                os.listdir(samples_path), 
                logger=logger, 
                desc='[LOAD] [FLAMBY - ISIC2019] ...resizing... ',
                total=len(os.listdir(samples_path)),
                mininterval=10,
                ):
                jobs.append(
                    workhorse.submit(
                        _resize_and_maintain, 
                        os.path.join(samples_path, sample_path), 
                        os.path.join(os.path.join(base_path, 'resized'), sample_path),
                        RESIZE
                    )
                )
            for job in concurrent.futures.as_completed(jobs):
                _ = job.result() 
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...resizing is completed!')

        ## remove raw files
        logger.info('[LOAD] [FLAMBY - ISIC2019] Remove raw files...!')
        shutil.rmtree(samples_path)
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...raw files are removed!')
    else:
        logger.info('[LOAD] [FLAMBY - ISIC2019] ...resizing has already been completed!')

    # assign datasets to clients
    logger.info('[LOAD] [FLAMBY - ISIC2019] Assign dataset by hopsital!')
    inputs_path = os.path.join(base_path, 'resized')
    client_datasets = []
    for hospital in HOSPITAL_NAME:
        client_datasets.append(_get_client_datasets(args, hospital, inputs_path, aggregated, TRAIN_TRANSFORM, TEST_TRANSFORM))
    logger.info('[LOAD] [FLAMBY - ISIC2019] ...assigned dataset!')
    
    # adjust arguments
    args.in_channels = 3
    args.num_classes = 8
    args.K = 6
    return {}, client_datasets, args
