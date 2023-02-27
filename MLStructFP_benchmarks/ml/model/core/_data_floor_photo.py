"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - CORE - FLOOR PHOTO

Photo data.
"""

__all__ = [
    '_SESSION_EXPORT_VERSION',
    'DataFloorPhoto',
    'load_floor_photo_data_from_session'
]

from MLStructFP.utils import DEFAULT_PLOT_DPI, configure_figure, make_dirs

from datetime import datetime
from typing import List, Dict, Tuple, Any
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time

_DATA_DTYPE: str = 'uint8'
_SESSION_EXPORT_VERSION: str = '1.0'


def _is_dict_equal(x: Dict[Any, Any], y: Dict[Any, Any]) -> bool:
    """
    Returns true if both dicts are equal.

    :param x: Dict
    :param y: Dict
    :return: True if equal
    """
    kx = list(x.keys())
    ky = list(y.keys())
    if len(kx) != len(ky):
        return False
    for i in range(len(kx)):
        if str(kx[i]) != str(ky[i]):  # Assert keys
            return False
        xki: Any = x[kx[i]]
        yki: Any = y[ky[i]]
        if isinstance(xki, (list, tuple)):
            assert isinstance(yki, (list, tuple))
            if len(xki) != len(yki):
                return False
            for j in range(len(xki)):
                if xki[j] != yki[j]:
                    return False
        else:
            if type(xki) != type(yki):
                return False
            if xki != yki:
                return False
    return True


class DataFloorPhoto(object):
    """
    Floor Photo data, which stores binary/photo from a given path. The path must contain images with the format:

    XXX_binary.npz
    XXX_photo.pnz

    Where XXX is a unique ID which represents the image package (floors), binary is the black/white image that
    represents wall rectangles, and photo is the processed patch from the real floor plan image.
    """
    _floor_photo_ch: int  # Number of photo channels
    _floor_photo_size: int  # Size of photo image
    _loaded_session: Dict[str, Any]  # Stores session data
    _parts: List[int]  # List of part IDs
    _path: str  # Path containg the images

    def __init__(self, path: str, shuffle_parts: bool = True) -> None:
        """
        Constructor.

        :param path: The path that stores the images. Requires all data to have the same image dimension
        :param shuffle_parts: If true, shuffles the part IDs
        """
        assert os.path.isdir(path), f'Path <{path}> does not exist'
        self._path = path
        self._parts = []
        self._loaded_session = {}
        for f in os.listdir(path):
            j = f.split('_')
            if os.path.splitext(f)[1] != '.npz':
                continue
            assert len(j) == 2, f'File require name format ID_type.npz, but <{f}> was found in path'
            assert j[0].isnumeric(), f'File ID must be numeric, however <{j[0]}> was provided'
            f_id = int(j[0])
            if f_id in self._parts:
                continue
            for img in self._get_file(f_id):
                assert os.path.isfile(img), f'File <{img}> was expected in path, but not found'
            self._parts.append(f_id)

        # Retrieve shape
        assert len(self._parts) > 0, 'No valid files were found at given path'
        self._parts.sort()
        s: Tuple[int, ...] = np.load(self._get_file(self._parts[0])[0])['data'][0].shape
        self._floor_photo_size = s[0]
        self._floor_photo_ch = 1 if len(s) == 2 else s[2]

        # If shuffle part IDs
        if shuffle_parts:
            random.shuffle(self._parts)
        else:
            self._parts.sort()

    def _get_file(self, part_id: int) -> Tuple[str, str]:
        """
        Return the file from a given part ID.

        :param part_id: Part ID
        :return: File binary/photo
        """
        return os.path.join(self._path, f'{part_id}_binary.npz'), os.path.join(self._path, f'{part_id}_photo.npz')

    def get_image_shape(self) -> Tuple[int, int, int]:
        """
        Get image shape. As rect image channels are converted to target floor photo channels
        this dimension and number of channels with be equal in rect/floor photo data.

        :return: Tuple
        """
        return self._floor_photo_size, self._floor_photo_size, self._floor_photo_ch

    @property
    def total_parts(self) -> int:
        """
        :return: Total number of parts
        """
        return len(self._parts)

    def load_part(self, part: int, shuffle: bool) -> Dict[str, 'np.ndarray']:
        """
        Load part and save into memory.

        :param part: Num part
        :param shuffle: Shuffle data order
        :return: Binary/Photo data. Images are within (0, 1) range
        """
        assert 1 <= part <= self.total_parts, f'Number of parts overflow, min:1, max:{self.total_parts}'
        f = self._get_file(self._parts[part - 1])
        img_b: 'np.ndarray' = np.load(f[0])['data']  # Binary
        img_p: 'np.ndarray' = np.load(f[1])['data']  # Photo

        # Convert type
        if img_b.dtype != _DATA_DTYPE:
            img_b = np.array(img_b, dtype=_DATA_DTYPE)
        if img_p.dtype != _DATA_DTYPE:
            img_p = np.array(img_p, dtype=_DATA_DTYPE)

        # Check length is the same
        s_b = img_b.shape
        s_p = img_p.shape
        assert s_b == s_p, \
            f'Part {part} image shape from binary/photo differs, value binary: {s_b}, photo: {s_p}'
        s = s_b

        # Assert shape for 1-channel image
        if len(s) == 3:
            img_b = img_b.reshape((*s, 1))
            img_p = img_p.reshape((*s, 1))

        # Shuffle
        if shuffle:
            indices = np.arange(img_b.shape[0])
            np.random.shuffle(indices)
            img_b = img_b[indices]
            img_p = img_p[indices]

        # Assemble output and return
        out = {
            'binary': img_b,
            'photo': img_p
        }
        gc.collect()
        return out

    @staticmethod
    def plot_image_example_id(
            part: Dict[str, 'np.ndarray'],
            imid: int,
            title: str = '',
            show: bool = True
    ) -> None:
        """
        Plot image from ID.

        :param part: Partition data
        :param imid: Image ID, from 0 to len(part)
        :param title: Optional image title
        :param show: Shows the image
        """
        if title == '':
            title = f'ID {imid}'

        kwargs = {'cfg_grid': False}
        fig = plt.figure(dpi=DEFAULT_PLOT_DPI)
        plt.title(title)
        # fig.subplots_adjust(hspace=.5)
        plt.axis('off')
        configure_figure()

        ax1: 'plt.Axes' = fig.add_subplot(121)
        ax1.title.set_text('Photo')
        ax1.imshow(part['photo'][imid], cmap='gray')
        plt.xlabel('x $(px)$')
        plt.ylabel('y $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)

        ax2 = fig.add_subplot(122)
        ax2.title.set_text('Binary')
        ax2.imshow(part['binary'][imid], cmap='gray')
        # plt.xlabel('x $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)

        if show:
            plt.show()

    def save_session(self, filename: str, description: str = '') -> None:
        """
        Save current session.

        :param filename: File to save the session
        :param description: Session description
        """
        filename = os.path.splitext(filename)[0]
        if '.json' not in filename:
            filename += '.json'
        make_dirs(filename)
        with open(filename, 'w', encoding='utf-8') as fp:
            data = {

                # Export version
                'version': _SESSION_EXPORT_VERSION,
                'class': 'DataFloorPhoto',
                'date_save': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                'description': description,

                # Basic data
                'parts': self._parts,
                'path': self._path,
                'floor_photo_ch': self._floor_photo_ch,
                'floor_photo_size': self._floor_photo_size,

            }
            json.dump(data, fp, indent=2)

            self._loaded_session = {
                'file': filename,
                'description': description
            }

        # Collect garbage
        gc.collect()

    def load_session(self, filename: str) -> None:
        """
        Load session from file.

        :param filename: Load file from file
        """
        filename = os.path.splitext(filename)[0]
        if '.json' not in filename:
            filename += '.json'
        with open(filename, 'r') as fp:
            data = json.load(fp)

            # Check version of the export is the same
            assert data['version'] == _SESSION_EXPORT_VERSION, \
                'Outdated session export version, needed {0}, current {1}'.format(_SESSION_EXPORT_VERSION, data['version'])

            # Check object data class is the same
            assert data['class'] == 'DataFloorPhoto', 'Data class is not valid'
            assert data['floor_photo_ch'] == self._floor_photo_ch, 'Floor image channels changed'
            assert data['floor_photo_size'] == self._floor_photo_size, 'Floor image size changed'

            # Check parts ID are the same
            assert len(data['parts']) == self.total_parts
            for i in data['parts']:
                assert i in self._parts, f'Part ID <{i}> does not exists'
            self._parts = data['parts']

            self._loaded_session = {
                'file': filename,
                'description': data['description']
            }

        # Collect garbage
        time.sleep(1)
        gc.collect()

    def update_session(self) -> None:
        """
        Updates session.
        """
        assert len(self._loaded_session.keys()) == 2, 'Session not loaded'
        print(f"Updating session <{self._loaded_session['file']}>")
        self.save_session(
            filename=self._loaded_session['file'],
            description=self._loaded_session['description']
        )


def load_floor_photo_data_from_session(filename: str) -> 'DataFloorPhoto':
    """
    Load data floor photo from session file.

    :param filename: Session file
    :return: Data
    """
    if '.json' not in filename:
        filename += '.json'
    assert os.path.isfile(filename), f'Session file <{filename}> does not exist'

    with open(filename, 'r') as fp:
        data = json.load(fp)
    assert data['version'] == _SESSION_EXPORT_VERSION, \
        'Outdated session export version, needed {0}, current {1}'.format(
            _SESSION_EXPORT_VERSION, data['version'])

    data = DataFloorPhoto(path=data['path'])
    data.load_session(filename=filename)

    return data
