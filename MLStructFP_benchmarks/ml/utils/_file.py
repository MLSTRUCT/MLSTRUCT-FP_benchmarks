"""
MLSTRUCTFP BENCHMARKS - ML - UTILS - FILE

File utility functions.
"""

__all__ = [
    'file_md5',
    'load_history_from_csv'
]

from typing import Dict, Any, List

import hashlib
import os


def file_md5(fname, buffer_size: int = 65536) -> str:
    """
    Returns md5 of a file.

    :param fname: File to compute hash
    :param buffer_size: Buffer size in bytes
    :return: File hash
    """
    assert buffer_size > 0
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(buffer_size), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_history_from_csv(file_csv: str) -> Dict[str, Any]:
    """
    Load train history from file.

    :param file_csv: File to load from
    :return: History dict
    """
    assert os.path.isfile(file_csv), f'CSV file <{file_csv}> does not exist'
    csv_file = open(file_csv, 'r')
    lines: List[str] = []
    for i in csv_file:
        lines.append(i.strip())
    assert len(lines) > 0, 'File empty'
    keys: List[str] = lines[0].split(',')
    assert keys.pop(0) == 'epoch', 'CSV file format invalid'
    assert len(keys) > 1, 'File invalid'
    history = {}
    for k in keys:
        history[k] = []
    for i in range(len(lines)):
        if i < 1:
            continue
        j = lines[i].split(',')
        j.pop(0)  # Pop epoch
        for w in range(len(j)):
            history[keys[w]].append(float(j[w]))
    csv_file.close()
    return history
