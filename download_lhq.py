"""
Based on https://github.com/wldhx/yadisk-direct repo
"""
import os
import math
import argparse
import requests
import shutil
import urllib
from urllib import parse as urlparse

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """
    Copy-pasted from https://stackoverflow.com/a/53877507/2685677
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file_with_progress(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


DATASET_TO_LINKS = {
    'lhq_full': [
        'https://disk.yandex.ru/d/5NNZHCqTjvOulw',
        'https://disk.yandex.ru/d/DMd5V-fcqAlxAg',
        'https://disk.yandex.ru/d/zBLgIqB6hROS9g',
        'https://disk.yandex.ru/d/hXcF8seffmAzxQ',
    ],
    'lhq_1024': [
        'https://disk.yandex.ru/d/EKtxZD-_MN4SrA',
        'https://disk.yandex.ru/d/5H4aGL6yuZFdVQ',
        'https://disk.yandex.ru/d/T1QW7kh27D_EYQ',
    ],
    'lhq_1024_jpg': [
        'https://disk.yandex.ru/d/Sz1gPiMoUregEQ',
    ],
    'lhq_256': [
        'https://disk.yandex.ru/d/HPEEntpLv8homg',
    ],
    'lhq_metadata': [
        'https://disk.yandex.ru/d/DOr5CP_QpZtGRQ',
    ],
}

API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'


def get_real_direct_link(sharing_link: str) -> str:
    pk_request = requests.get(API_ENDPOINT.format(sharing_link))

    return pk_request.json()['href']


def download_file(url: str, save_path: str):
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def convert_size(size_bytes: int):
    """
    Copy-pasted https://stackoverflow.com/a/14822210/2685677
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])


def download_dataset(dataset_name: str):
    assert dataset_name in list(DATASET_TO_LINKS.keys()), \
        f"Wrong dataset name. Possible options are: {', '.join(DATASET_TO_LINKS.keys())}"
    file_urls = DATASET_TO_LINKS[dataset_name]

    save_dir = f'lhq/{dataset_name}'
    print(f'Saving files into ./{save_dir} directory...')
    os.makedirs(save_dir, exist_ok=True)

    for i, file_url in enumerate(file_urls):
        download_url = get_real_direct_link(file_url)
        url_parameters = urlparse.parse_qs(urlparse.urlparse(download_url).query)
        filename = url_parameters['filename'][0]
        file_size = convert_size(int(url_parameters['fsize'][0]))
        save_path = os.path.join(save_dir, filename)
        print(f'Downloading {i+1}/{len(file_urls)} files: {save_path} (size: {file_size})')
        download_file_with_progress(download_url, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Landscapes HQ dataset downloader")
    parser.add_argument('dataset', type=str, choices=list(DATASET_TO_LINKS.keys()))
    args = parser.parse_args()
    download_dataset(args.dataset)
