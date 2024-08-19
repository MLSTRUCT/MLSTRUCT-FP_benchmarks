"""
MLSTRUCT-FP BENCHMARKS - ML - UTILS - ARRAY

Array utils.
"""

__all__ = ['get_key_hash']

from typing import List


def get_key_hash(*args) -> str:
    """
    Returns key from values.

    :param args: Get data from values
    :return: Key
    """
    key: List[str] = []
    for k in args:
        if isinstance(k, list):
            newk = []
            for w in k:
                if isinstance(w, (int, float, str)):
                    newk.append(str(w))
            key.append('|'.join(newk))
        elif k is None:
            continue
        elif isinstance(k, (int, float, str)):
            key.append(k)
    for i in range(len(key)):
        key[i] = str(key[i]).replace('_', '')
    return '_'.join(key)
