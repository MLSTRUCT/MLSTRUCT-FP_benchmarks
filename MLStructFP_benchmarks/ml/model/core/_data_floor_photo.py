"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - CORE - FLOOR PHOTO

Photo data.
"""

__all__ = [
    '_SESSION_EXPORT_VERSION',
    'DataFloorPhoto',
    'load_floor_photo_data_from_session'
]

from MLStructFP.utils import DEFAULT_PLOT_DPI, configure_figure, make_dirs

from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import zipfile

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


# noinspection PyUnresolvedReferences
def _npz_headers(npz: str):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            # noinspection PyProtectedMember
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype


class DataFloorPhoto(object):
    """
    Floor Photo data, which stores binary/photo from a given path. The path must contain images with the format:

    XXX_binary.npz
    XXX_photo.pnz

    Where XXX is a unique ID that represents the image package (floors), binary is the black/white image that
    represents wall rectangles, and the photo is the processed patch from the real floor plan image.
    """
    _floor_photo_ch: int  # Number of photo channels
    _floor_photo_size: int  # Size of photo image
    _loaded_session: Dict[str, Any]  # Stores session data
    _parts: List[int]  # List of part IDs
    _path: str  # Path containg the images
    _split: Optional[List[List[int]]]  # Train/test split

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
        self._split = []
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
        Get image shape. As rect image channels are converted to target floor photo channels,
        this dimension and number of channels must be equal in rect/floor photo data.

        :return: Tuple
        """
        return self._floor_photo_size, self._floor_photo_size, self._floor_photo_ch

    @property
    def total_parts(self) -> int:
        """
        :return: Total number of parts
        """
        return len(self._parts) if len(self._split) == 0 else 1

    @property
    def total_images(self) -> int:
        """
        :return: Number of total images
        """
        total = 0
        for i in self._parts:  # Iterate each part
            i_info = list(_npz_headers(self._get_file(i)[0]))[0]  # ('data', (N, SIZE, SIZE), dtype('DTYPE'))
            total += i_info[1][0]
        return total

    def load_part(self, part: int, shuffle: bool = False, ignore_split: bool = False) -> Dict[str, 'np.ndarray']:
        """
        Load part and save into memory.

        :param part: Num part. If split, one returns train, else, return test
        :param shuffle: Shuffle data order
        :param ignore_split: If true, ignores train/test split
        :return: Binary/Photo data. Images are within (0, 1) range
        """
        img_b: 'np.ndarray'
        img_p: 'np.ndarray'
        if len(self._split) == 0 or ignore_split:
            assert 1 <= part <= len(self._parts), f'Number of parts overflow, min:1, max:{len(self._parts)}'
            f = self._get_file(self._parts[part - 1])
            img_b = np.load(f[0])['data']  # Binary
            img_p = np.load(f[1])['data']  # Photo
        else:
            assert part in (1, 2), '1 returns train, 2 test. No other part value allowed'
            # First, get all images size and create a numpy zero object
            imgs = 0  # Total images loaded so far
            sizes: Dict[int, int] = {}  # Size for each part
            for i in self._split[part - 1]:  # Iterate loaded parts
                i_info = list(_npz_headers(self._get_file(i)[0]))[0]  # ('data', (N, SIZE, SIZE), dtype('DTYPE'))
                i_shp = i_info[1]
                assert i_shp[1] == i_shp[2] == self._floor_photo_size, \
                    'Each image part must have size ({0}, {0})'.format(self._floor_photo_size)
                assert i_info[2] == _DATA_DTYPE, \
                    f'Data type does not match, requires {_DATA_DTYPE}, but {i_info[2]} was provided when loading parts'
                imgs += i_shp[0]
                sizes[i] = i_shp[0]

            # Create empty numpy shape
            new_shape = (imgs, self._floor_photo_size, self._floor_photo_size)
            if self._floor_photo_ch != 1:
                new_shape = (*new_shape, self._floor_photo_size)
            img_b = np.zeros(new_shape, dtype=_DATA_DTYPE)
            img_p = np.zeros(new_shape, dtype=_DATA_DTYPE)

            j = 0  # Index to add
            k = 0  # Number of processed parts
            for i in self._split[part - 1]:  # Iterate loaded parts
                f = self._get_file(i)
                img_b[j:j + sizes[i]] = np.load(f[0])['data']  # Binary
                img_p[j:j + sizes[i]] = np.load(f[1])['data']  # Photo
                j += sizes[i]
                k += 1
                if k % 50 == 0:
                    gc.collect()

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

    def assemble_train_test(self, split: float) -> 'DataFloorPhoto':
        """
        Assemble train/test data.

        :param split: Split percentage images in train/test
        :return: Self
        """
        assert 0 < split < 1, 'Split must be between 0 and 1'
        train = []
        test = []
        for i in range(int(split * len(self._parts))):
            train.append(self._parts[i])
        for i in self._parts:
            if i not in train:
                test.append(i)
        self._split = [train, test]
        return self

    @property
    def train_split(self) -> float:
        """
        :return: Split partition percentage
        """
        if len(self._split) == 0:
            return 0
        return round(len(self._split[0]) / len(self._parts), 2)

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
                'floor_photo_ch': self._floor_photo_ch,
                'floor_photo_size': self._floor_photo_size,
                'parts': self._parts,
                'path': self._path,
                'split': self._split,

            }
            # noinspection PyTypeChecker
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

            # Check if the version of the export is the same
            assert data['version'] == _SESSION_EXPORT_VERSION, \
                'Outdated session export version, needed {0}, current {1}'.format(_SESSION_EXPORT_VERSION, data['version'])

            # Check object data class is the same
            assert data['class'] == 'DataFloorPhoto', 'Data class is not valid'
            assert data['floor_photo_ch'] == self._floor_photo_ch, 'Floor image channels changed'
            assert data['floor_photo_size'] == self._floor_photo_size, 'Floor image size changed'

            # Check parts ID are the same
            assert len(data['parts']) == len(self._parts)
            for i in data['parts']:
                assert i in self._parts, f'Part ID <{i}> does not exists'
            self._parts = data['parts']
            self._split = data['split']

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

    @property
    def filename(self) -> str:
        """
        Return the filename of the saved session.
        """
        if len(self._loaded_session.keys()) != 2:
            return ''
        return os.path.splitext(os.path.basename(self._loaded_session['file']))[0]


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
