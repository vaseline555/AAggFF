import os
import sys
import urllib
import logging
import torchvision

logger = logging.getLogger(__name__)

__all__ = ['accept_license', 'download_data']



def accept_license(license_link, dataset_name):
    """This function forces the user to accept the license terms before proceeding with the download.
    (Adapted from https://github.com/owkin/FLAMBY/blob/main/FLAMBY/utils.py#L253)
    """
    while True:
        answer = input(
            f"\n\t\t [FLAMBY - {dataset_name.upper()}] Have you taken the time to read and accept the data terms on the original website, available at the following link: "
            f"{license_link} ? | (Y/N) \n\n"
        )
        if any(answer.lower() == f for f in ['yes', 'y']):
            logger.info(f"[LOAD] [{dataset_name.upper()}] You ACCEPTED the license agreement...")
            logger.info(f"[LOAD] [{dataset_name.upper()}] You may now proceed to download!\n")
            break

        elif any(answer.lower() == f for f in ['no', 'n']):
            logger.info(
                f"[LOAD] [{dataset_name.upper()}] Since you have not read and accepted the license terms the download of the dataset is aborted." 
                "Please come back when you have fulfilled this legal obligation."
            )
            sys.exit(1)
        else:
            logger.info(
                f"[LOAD] [{dataset_name.upper()}] If you wish to proceed with the download you need to read and accept the license and data terms of the original data owners."
                "Please read and accept the terms and answer yes.\n\n\n"
            )

def download_data(download_root, dataset_name, url_dict, md5_dict):
    """Download data and extract if it is archived.
    """
    # download data from web
    logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] Start downloading data...!')
    try:
        for (name, md5) in zip(url_dict.keys(), md5_dict.keys()):
            if any(ext in url_dict[name] for ext in ['.bz2', '.gz', '.tar', '.tbz', '.tbz2', '.tgz', '.xz', '.zip']):
                torchvision.datasets.utils.download_and_extract_archive(
                    url=url_dict[name], 
                    download_root=download_root, 
                    md5=md5_dict[md5], 
                    remove_finished=True
                )
            else:
                try:
                    with urllib.request.urlopen(urllib.request.Request(url_dict[name])) as response:
                        from tqdm import tqdm
                        with open(os.path.join(download_root, url_dict[name].split('/')[-1]), 'wb') as fh, tqdm(total=response.length) as pbar:
                            for chunk in iter(lambda: response.read(1024 * 32), b""):
                                # filter out keep-alive new chunks
                                if not chunk: continue
                                fh.write(chunk)
                                pbar.update(len(chunk))
                except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
                    if url_dict[name][:5] == 'https':
                        url = url_dict[name].replace('https:', 'http:')
                        with urllib.request.urlopen(urllib.request.Request(url)) as response:
                            from tqdm import tqdm
                            with open(os.path.join(download_root, url_dict[name].split('/')[-1]), 'wb') as fh, tqdm(total=response.length) as pbar:
                                for chunk in iter(lambda: response.read(1024 * 32), b""):
                                    # filter out keep-alive new chunks
                                    if not chunk: continue
                                    fh.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        logger.exception(e)
                        raise Exception(e)
        else:
            logger.info(f'[LOAD] [FLAMBY - {dataset_name.upper()}] ...finished downloading data!')
    except Exception as e:
        logger.exception(e)
        raise Exception(e)
