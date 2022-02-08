import os

import urllib3
import socket
import cfg
import cv2
import time
import re
import cv_utils
import http
from datetime import datetime
from contextlib import contextmanager
import numpy as np


def trigger_remote_camera(img_number=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    sock.sendto(bytes(f'F', "utf-8"), (cfg.ip, cfg.udp_port))
    time.sleep(cfg.autofocus_time_seconds)  # wait for autofocus
    sock.sendto(bytes(f'C{img_number}', "utf-8"), (cfg.ip, cfg.udp_port))
    pass


def filename_2_number(name):
    splitted = name.split('_')
    if len(splitted) == 2:
        return -1
    else:
        num = splitted[-1].split('.')[0]
        return int(num)


def find_photo(img_number=0):  # todo check status?
    response = urllib3.PoolManager().request("GET", f'{cfg.ip}:{cfg.http_port}')
    if response.status == http.HTTPStatus.OK:
        response = str(response.data)
    else:
        return None
    pattern = re.compile(f'>{cfg.img_prefix}{img_number}(?:_\d+)?\.jpg')
    matches = pattern.findall(response)
    if not matches:
        return None
    matches = sorted([m[1:] for m in matches], key=filename_2_number)
    filename = sorted(matches, key=filename_2_number)[-1]
    return filename


def download_photo(img_number=0, last_filename=None):
    filename_on_server = find_photo(img_number)
    if filename_on_server == last_filename:
        return None
    url = f'{cfg.ip}:{cfg.http_port}/{filename_on_server}'
    response = urllib3.PoolManager().request("GET", url)
    data = response.data if response.status == http.HTTPStatus.OK else None
    return data


def take_photo(save_path=cfg.camera_save_path, img_number_on_server=0):
    last_filename = find_photo(img_number_on_server)
    trigger_remote_camera(img_number_on_server)
    i, delay, timeout = 0, 0.3, 10
    img_data = download_photo(img_number_on_server, last_filename)
    while img_data is None:
        if i * delay >= timeout:
            print('failed to take photo')
            return False
        i += 1
        time.sleep(delay)
        img_data = download_photo(img_number_on_server, last_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(img_data)
    return True

@contextmanager
def measure_time():
    t0 = None
    try:
        t0 = datetime.now()
        yield
    finally:
        t1 = datetime.now()
        if t0 is not None:
            delta = t1 - t0
            print(f'execution time: {delta.total_seconds()}')


if __name__ == '__main__':
    filename = cfg.camera_save_path
    with measure_time():
        take_photo(filename)
    img = np.rot90(cv2.imread(filename))
    cv_utils.display_img(img, scale=10)


