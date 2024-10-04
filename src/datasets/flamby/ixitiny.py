import os
import torch
import monai
import random
import logging

import nibabel as nib

logger = logging.getLogger(__name__)



class IXITiny(torch.utils.data.Dataset):
    def __init__(self, hospital_name, img_path, label_path, transform=None, target_transform=None):
        self.name = hospital_name
        self.img_path = img_path
        self.label_path = label_path

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img = nib.load(self.img_path[idx]).get_fdata()
        label = nib.load(self.label_path[idx]).get_fdata()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img.float(), label

    def __len__(self):
        return len(self.img_path)

    def __repr__(self):
        return self.name

# helper method to fetch IXI-Tiny brain MRI segmentation dataset
def fetch_dataset(args, root):
    HOSPITAL_NAME = ['Guys', 'HH', 'IOP']
    COMMON_SHAPE = (48, 60, 48)
    MODALITY = 'T1'
    INPUT_TRANSFORM = monai.transforms.Compose([
        monai.transforms.ToTensor(), 
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'), 
        monai.transforms.Resize(COMMON_SHAPE), # default transforms
        monai.transforms.NormalizeIntensity(), # intensity transform
    ])
    TARGET_TRANSFORM = monai.transforms.Compose([
        monai.transforms.ToTensor(), 
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'), 
        monai.transforms.Resize(COMMON_SHAPE), # default transforms
        monai.transforms.AsDiscrete(to_onehot=2) # one-hot transform
    ])

    def _get_paths(samples_path, modality):
        files = [f for f in os.listdir(samples_path) if os.path.isdir(os.path.join(samples_path, f))]

        path_map = dict()
        for hospital in HOSPITAL_NAME:
            file_name = [file for file in files if hospital in file]
            paths = {
                hospital: (
                    [os.path.join(samples_path, file, modality, f'{file}_image.nii.gz') for file in file_name],
                    [os.path.join(samples_path, file, 'label', f'{file}_label.nii.gz') for file in file_name]
                )
            }
            path_map.update(paths)
        return path_map

    def _get_client_datasets(args, hospital, path_map, input_transfrom, target_transform):
        def __get_train_test_indices(args, array):
            indices = list(range(len(array)))
            random.seed(args.seed); random.shuffle(indices)
            test_size = int(len(indices) * args.test_size)
            return indices[test_size:], indices[:test_size]

        raw_inputs, raw_targets = path_map[hospital]
        train_indices, test_indices = __get_train_test_indices(args, raw_inputs)

        train_inputs, test_inputs = [raw_inputs[i] for i in train_indices], [raw_inputs[i] for i in test_indices]
        train_targets, test_targets = [raw_targets[i] for i in train_indices], [raw_targets[i] for i in test_indices]
        return (
            IXITiny(f'[IXITiny] CLIENT < {hospital} > (train)', train_inputs, train_targets, input_transfrom, target_transform),
            IXITiny(f'[IXITiny] CLIENT < {hospital} > (test)', test_inputs, test_targets, input_transfrom, target_transform)
        )

    # create client-path hashmap
    logger.info('[LOAD] [FLAMBY - IXITiny] Create index hashmap and identifier hashmap...!')
    samples_path = os.path.join(root, 'ixitiny', '7kd5wj7v7p-3', 'IXI_sample')
    path_map = _get_paths(samples_path, MODALITY)
    logger.info(f'[LOAD] [FLAMBY - IXITiny] ...successfully created hashmaps!')

    # assign datasets to clients
    logger.info('[LOAD] [FLAMBY - IXITiny] Assign dataset by hopsital!')
    client_datasets = []
    for hospital in HOSPITAL_NAME:
        client_datasets.append(_get_client_datasets(args, hospital, path_map, INPUT_TRANSFORM, TARGET_TRANSFORM))
    logger.info('[LOAD] [FLAMBY - IXITiny] ...assigned dataset!')

    # adjust arguments
    args.in_channels = 1
    args.num_classes = 2
    args.K = 3
    return {}, client_datasets, args