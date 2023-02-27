"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - CORE - DATA X/Y

Model data x/y.
"""

__all__ = [
    'ModelDataXY',
    '_SESSION_EXPORT_VERSION'
]

from MLStructFP_benchmarks.ml.utils import file_md5, scale_array_to_range
from MLStructFP_benchmarks.ml.utils.plot import DataXYPlot

from datetime import datetime
from pandas.util import hash_pandas_object
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from typing import List, Union, Tuple, Dict, Optional
import copy
import gc
import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import random
import time

_SESSION_EXPORT_VERSION: str = '1.5'
_TRAIN_TEST_ERROR: str = 'Train test has not been split yet, please use data.save_train_test() method'
_VALID_COLUMN: List[str] = ['x', 'y', 'xy', 'yx']
_VALID_XY: List[str] = ['xy', 'yx']
_XY_ERROR: str = 'xy param must be "x", "y" or "xy'


def _load_literals(literals_file: str) -> Dict[str, List[str]]:
    """
    Load a literals file and return a dict.

    :param literals_file: File to load
    :return: Dict
    """
    fl = open(literals_file, 'r')
    d = {}
    for i in fl:
        line = i.strip().split(',')
        if len(line) <= 1:
            continue
        key = line[0]
        line.pop(0)
        value = ','.join(line)
        value = value.replace('"', '')
        if key in d.keys():
            raise ValueError(f'Repeated <{key}> key')
        d[key] = value.strip('][').split(', ')
        assert isinstance(d[key], list), f'Literal at key <{key}> must be a list'
        assert len(d[key]) > 0, f'Literal at key <{key}> cannot be empty'
        for j in range(len(d[key])):
            d[key][j] = str(d[key][j]).replace("'", '')
    return d


def _train_test_split_project_id(
        x: 'pd.DataFrame',
        y: 'pd.DataFrame',
        test_size: float,
        id_column: str
) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray']:
    """
    Split dataset in tran and test.

    :param x: X dataset
    :param y: Y dataset
    :param test_size: Test size
    :param id_column: Column ID
    :return: Split data in the format x_train, x_test, y_train, y_test
    """
    assert 1 > test_size > 0, \
        'test size must be between 0 and 1'
    assert x[id_column].count() == y[id_column].count(), \
        'data must have the same number of files'
    x_id: 'pd.DataFrame' = x[id_column]

    test_len = int(test_size * x_id.count())
    available_ids = x_id.unique()
    random.shuffle(available_ids)

    # Generate fake mask
    id_mask = x_id == max(available_ids) + 1

    # Selects ID upon reaching 'test_len' elements
    selected_ids = []
    total_added = 0
    for _id in available_ids:
        mask_id = x_id == _id
        count_id = sum(mask_id)
        total_added += count_id
        id_mask = id_mask | mask_id
        selected_ids.append(_id)
        if total_added > test_len:
            break

    assert sum(id_mask) == total_added, \
        'ID mask does not sum as the total added'

    # Filter new x and y datasets
    x_test = x[id_mask]
    y_test = y[id_mask]
    id_test = np.random.permutation(len(x_test))
    x_test = x_test.iloc[id_test]
    y_test = y_test.iloc[id_test]

    # Get Train
    x_train = x[~id_mask]
    y_train = y[~id_mask]
    id_train = np.random.permutation(len(x_train))
    x_train = x_train.iloc[id_train]
    y_train = y_train.iloc[id_train]

    # Return values
    return x_train.values, x_test.values, y_train.values, y_test.values


class ModelDataXY(object):
    """
    Data from two sources, X and Y comes from two different files.
    """
    _column_renames: List[Tuple[str, str]]
    _data_x: 'pd.DataFrame'
    _data_x_checkpoint: 'pd.DataFrame'
    _data_y: 'pd.DataFrame'
    _data_y_checkpoint: 'pd.DataFrame'
    _drop_cols: Dict[str, List[str]]
    _drop_id: bool
    _filename: str
    _filepath: str
    _id: List[str]  # Name of columns
    _image_col: str
    _images_hash: Tuple[str, str]
    _images_x: 'np.ndarray'  # Source images in x
    _images_y: 'np.ndarray'  # Source images in y
    _is_split_train_test: bool
    _literals: Dict[str, List[str]]
    _load_images: bool
    _loaded_session = Dict[str, str]
    _minmax_scaler: 'preprocessing.MinMaxScaler'
    _minmax_scaler_applied: bool
    _raw_data_x: 'pd.DataFrame'
    _raw_data_y: 'pd.DataFrame'
    _scaler_col: List[str]
    _split_train_test_mode: str
    _train_test: Dict[str, Union['pd.DataFrame', 'np.ndarray']]  # Stores data from train test split

    plot: 'DataXYPlot'

    def __init__(
            self,
            filename: str,
            column_id: Union[List[str], str],
            image_col: str = '',
            image_size: int = 64,
            drop_id: bool = True,
            path: Union[str, Path] = ''
    ) -> None:
        """
        Constructor.

        :param filename: Data file
        :param column_id: Name of columns id, These columns are not affected by any scaler
        :param image_col: Column image ID, if none, don't load the images
        :param image_size: Image size to load (px)
        :param drop_id: Mark as dropped all columns ID
        :param path: Path to load from
        """
        # Check if path exists
        if isinstance(path, Path):
            max_path = 0
            while True:
                filex = str(path) + os.path.sep + os.path.splitext(filename)[0] + '_x.csv'
                if os.path.isfile(filex):
                    break
                path = path.parent
                max_path += 1
                if max_path > 50:
                    break

        self._filepath = str(path)
        if self._filepath != '':
            self._filepath += os.path.sep
        self._filename = os.path.splitext(filename)[0]

        fileloc = os.path.join(self._filepath, self._filename)
        file_x = fileloc + '_x.csv'
        file_y = fileloc + '_y.csv'
        file_literals = fileloc + '_literals.csv'
        file_metrics = fileloc + '_metrics.csv'

        # Check file exists
        assert os.path.isfile(file_x), \
            f'X file <{file_x}> does not exist in path <{self._filepath}>'
        assert os.path.isfile(file_y), \
            f'Y file <{file_y}> does not exist in path <{self._filepath}>'
        assert os.path.isfile(file_literals), \
            f'Literals file <{file_literals}> does not exist in path <{self._filepath}>'
        assert os.path.isfile(file_metrics), \
            f'Metrics file <{file_metrics}> does not exist in path <{self._filepath}>'

        self._raw_data_x = pd.read_csv(file_x)
        self._raw_data_y = pd.read_csv(file_y)
        self._literals = _load_literals(file_literals)
        self._metrics = pd.read_csv(file_metrics)

        # Operative data
        self._drop_cols = {'x': [], 'y': []}
        self._inverse_scaler_warning = False
        self._loaded_session = {}
        self._load_images = True
        self._scaler_col_x = []
        self._scaler_col_y = []

        self._reset_all()

        # Save id
        if not isinstance(column_id, list):
            column_id = [column_id]
        if len(column_id) == 0:
            raise ValueError('column_id cannot be empty')
        if drop_id:
            self.set_drop(*column_id, axis='xy')
        self._drop_id = drop_id
        self._id = column_id

        # Save image ID
        if image_col != '':
            assert isinstance(image_size, int), 'Image size must be an integer'
            assert image_col in self._id, f'Column image <{image_col}> must be in column_id'
            assert image_size > 0, 'Image size must be greater than zero'
            assert math.log(image_size, 2).is_integer(), 'Image size must be a power of 2'

            # Load images
            file_images_x = fileloc + f'_images_x_{image_size}.npz'
            file_images_y = fileloc + f'_images_y_{image_size}.npz'
            assert os.path.isfile(file_images_x), \
                f'X images file <{file_images_x}> does not exist in path <{self._filepath}>'
            assert os.path.isfile(file_images_y), \
                f'Y images file <{file_images_y}> does not exist in path <{self._filepath}>'

            fimg_x = np.load(file_images_x)  # compressed file
            fimg_y = np.load(file_images_y)

            self._images_x = fimg_x['data']
            self._images_y = fimg_y['data']
            self._images_hash = (file_md5(file_images_x), file_md5(file_images_y))

            # Assert shape
            imgshape_x = np.shape(self._images_x[0])
            imgshape_y = np.shape(self._images_y[0])

            assert len(self._images_x) == len(self._images_y), \
                'Image size from x/y is not the same'
            assert len(self._images_x) == len(self._data_x), \
                'Image length must be the same as the data'
            assert len(imgshape_x) == len(imgshape_y), \
                'Number of dimensions of the image shape cannot be different from x and y'
            assert 3 <= len(np.shape(self._images_x)) <= 4, \
                'Image cannot have more than 4 dimensions, and less than 3'
            assert imgshape_x[0] == imgshape_x[1], \
                'Image shape must be the same'
            assert imgshape_x[0] == imgshape_y[0], \
                'Image shape must be the same as x and y'
            assert imgshape_x[0] == image_size, \
                'Image shape must be {0}x{0}, not {1}x{1}'.format(image_size, imgshape_x[0])

            del fimg_x, fimg_y
        else:
            self._images_x = np.ndarray((0, 0))
            self._images_y = np.ndarray((0, 0))

        self._image_col = image_col
        self._image_size = image_size

        # Create checkpoint
        self.checkpoint()

        # Create plot
        self.plot = DataXYPlot(self)
        self._column_renames = []

    def rename_cols(self, col: Union[Tuple[str, str], List[Tuple[str, str]]]) -> None:
        """
        Rename datasets cols.

        :param col: Source, Target names
        """
        if self._is_split_train_test:
            raise RuntimeError('Train test already split')
        if self._minmax_scaler_applied:
            raise RuntimeError('Scaler already applied')
        if not isinstance(col, list):
            col = [col]
        for i in range(len(col)):
            assert isinstance(col[i], tuple), f'Element {i} must be a tuple of 2 strings'
        source_cols: List[str] = []
        target_cols: List[str] = []
        for c in col:
            assert len(c) == 2
            assert isinstance(c[0], str) and isinstance(c[1], str)
            if c[0] == c[1]:
                continue
            assert c not in self._column_renames, 'Column already renamed'
            assert c[0] in self._data_x.columns, f'Source column <{c[0]}> does not exist'
            assert c[1] not in self._data_x.columns, f'Target column <{c[1]}> already exists in dataframe'
            assert c[0] not in source_cols, f'Source column <{c[0]}> already being renamed'
            assert c[1] not in target_cols, f'Target column <{c[1]}> already being renamed'
            source_cols.append(c[0])
            target_cols.append(c[1])
        for c in col:
            if c[0] == c[1]:
                continue
            self._column_renames.append(c)
            self._rename_column(c)

    def _rename_column(self, col: Tuple[str, str]) -> None:
        """
        Rename column.

        :param col: Source, Target names
        """
        self._data_x = self._data_x.rename(columns={col[0]: col[1]})
        self._data_y = self._data_y.rename(columns={col[0]: col[1]})

    def get_filepath(self) -> str:
        """
        Returns data file path.

        :return: File path
        """
        return self._filepath

    def get_filename(self) -> str:
        """
        :return: Returns data file name
        """
        return self._filename

    def get_data_hash(self, xy: str) -> str:
        """
        Returns data hash from x/y dataframes.

        :param xy: Which dataframe
        :return: Hash as str
        """
        _x: 'pd.DataFrame'
        _y: 'pd.DataFrame'
        if xy == 'data':
            _x = self.get_dataframe('xdata')
            _y = self.get_dataframe('ydata')
        elif xy == 'xy':
            if not self._is_split_train_test:
                return '0'
            _x = self.get_dataframe('x')
            _y = self.get_dataframe('y')
        elif xy == 'train':
            if not self._is_split_train_test:
                return '0'
            _x = self.get_dataframe('xtrain')
            _y = self.get_dataframe('ytrain')
        elif xy == 'test':
            if not self._is_split_train_test:
                return '0'
            _x = self.get_dataframe('xtest')
            _y = self.get_dataframe('ytest')
        elif xy == 'images':
            if self._image_col == '':
                return '0'
            h = hashlib.md5()
            h.update(self._images_hash[0].encode())
            h.update(self._images_hash[1].encode())
            return h.hexdigest()
        else:
            raise ValueError('xy must be "data", "xy", "train" or "test", "images"')
        return str(abs(int(hash_pandas_object(_x).sum())) + abs(int(hash_pandas_object(_y).sum())))

    def save_session(self, filename: str, unique_id_col: str, description: str = '') -> None:
        """
        Save current session.

        :param filename: File to save the session
        :param unique_id_col: Unique save file
        :param description: Session description
        """
        assert unique_id_col in self._id, f'Column {unique_id_col} not in ID'
        filename = os.path.splitext(filename)[0]
        if '.json' not in filename:
            filename += '.json'
        with open(filename, 'w', encoding='utf-8') as fp:
            def get_id(xy: str) -> List[int]:
                """
                Get ID list from test/train dataframes.

                :param xy: Name of dataframe.
                """
                if not self._is_split_train_test:
                    return []
                df = self.get_dataframe(xy, True)[[unique_id_col]]
                return df.values.reshape(1, -1)[0].tolist()

            shape = self._shape()
            data = {

                # Export version
                'version': '1.5',
                'class': 'ModelDataXY',
                'date_save': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                'session_unique_id_col': unique_id_col,
                'description': description,

                # Basic data
                'col_renames': self._column_renames,
                'drop_col': self._drop_cols,
                'drop_id': self._drop_id,
                'filename': self._filename,
                'id_col': self._id,
                'image_col': self._image_col,
                'image_size': self._image_size,
                'min_max_applied': self._minmax_scaler_applied,

                # Hashes
                'hash_data': self.get_data_hash('data'),
                'hash_images': self.get_data_hash('images'),
                'hash_test': self.get_data_hash('test'),
                'hash_train': self.get_data_hash('train'),
                'hash_xy': self.get_data_hash('xy'),

                # Shape
                'x_shape': shape['x'],
                'y_shape': shape['y'],
                'image_x_shape': self._images_x.shape,
                'image_y_shape': self._images_y.shape,

                # Store splits
                'train_test_split_mode': self._split_train_test_mode,
                'train_test_is_split': self._is_split_train_test,
                'train_id': get_id('xtrain'),
                'test_id': get_id('xtest')
            }

            json.dump(data, fp, indent=2)
            self._loaded_session = {
                'file': filename,
                'description': description,
                'unique_id_col': unique_id_col
            }

        # Collect garbage
        gc.collect()

    def _shape(self) -> Dict[str, List[int]]:
        """
        :return: Returns the shape of the data
        """
        return {
            'x': [len(self._data_x), len(self._data_x.columns)],
            'y': [len(self._data_y), len(self._data_y.columns)]
        }

    def load_session(self, filename: str, check_hash: bool = True) -> None:
        """
        Load session from file.

        :param filename: Load file from file
        :param check_hash: Checks file hash
        """
        if self._is_split_train_test:
            raise RuntimeError('Train test split, session cannot be loaded')
        if self._minmax_scaler_applied:
            raise RuntimeError('Scaler already applied, session cannot be loaded')
        assert isinstance(check_hash, bool)
        filename = os.path.splitext(filename)[0]
        if '.json' not in filename:
            filename += '.json'
        with open(filename, 'r') as fp:
            data = json.load(fp)

            # Check version of the export is the same
            assert data['version'] == _SESSION_EXPORT_VERSION, \
                'Outdated session export version, needed {0}, current {1}'.format(_SESSION_EXPORT_VERSION,
                                                                                  data['version'])

            # Check object data class is the same
            assert data['class'] == 'ModelDataXY', 'Data class is not valid'

            # Assert image has the same shape
            if self._load_images:
                assert self._images_x.shape[0] == data['image_x_shape'][0], 'X image length is not the same'
                assert self._images_y.shape[0] == data['image_y_shape'][0], 'Y image length is not the same'

            self._drop_cols = data['drop_col']
            self._column_renames = data['col_renames']
            for col in self._column_renames:
                self._rename_column(col)

            if data['min_max_applied']:
                self.scale_minmax()
                # pass

            # Check hash after scaling and column renaming is the same
            if check_hash:
                assert self.get_data_hash('data') == data['hash_data'], 'Data hash is not the same'
                if self._load_images:
                    assert self._image_col == data['image_col'], 'Image column ID is not the same'
                    assert self._image_size == data['image_size'], 'Image size must be the same'

            # Split train test
            if data['train_test_is_split']:
                id_np = {
                    'train_id': np.array(data['train_id']),
                    'test_id': np.array(data['test_id'])
                }

                self.save_train_test(
                    test_size=0,
                    split_by_session_id=id_np,
                )
                self._split_train_test_mode = data['train_test_split_mode']

                # Check hash
                if check_hash:
                    assert self.get_data_hash('xy') == data['hash_xy'], 'XY hash changed'
                    assert self.get_data_hash('train') == data['hash_train'], 'Train hash changed'
                    assert self.get_data_hash('test') == data['hash_test'], 'Test hash changed'

            # Remove train test
            else:
                self.remove_train_test()

            self._loaded_session = {
                'file': filename,
                'description': data['description'],
                'unique_id_col': data['session_unique_id_col']
            }

        # Collect garbage
        time.sleep(1)
        gc.collect()

    def update_session(self) -> None:
        """
        Updates session.
        """
        assert len(self._loaded_session.keys()) == 3, 'Session not loaded'
        print('Updating session <{0}> by <{1}>'.format(self._loaded_session['file'],
                                                       self._loaded_session['unique_id_col']))
        self.save_session(
            filename=self._loaded_session['file'],
            unique_id_col=self._loaded_session['unique_id_col'],
            description=self._loaded_session['description']
        )

    def get_dataframe(self, xy: str, get_id: bool = False, drop: bool = False) -> 'pd.DataFrame':
        """
        Returns dataframe.

        :param xy: Which dataframe
        :param get_id: Returns ID of the dataframe
        :param drop: Drop data from .get_drop()
        :return: Data frame
        """
        if get_id and drop:
            raise ValueError('ID and DROP cannot be True at the same time')
        df: 'pd.DataFrame'
        xy = xy.lower()
        if xy == 'xdata':
            df = self._data_x.copy()
        elif xy == 'ydata':
            df = self._data_y.copy()
        elif xy in ['x', 'y', 'xtest', 'ytest', 'xtrain', 'ytrain']:
            assert self._is_split_train_test, _TRAIN_TEST_ERROR
            df = self._train_test[xy].copy()
        else:
            raise ValueError('xy param must be "xdata", "ydata", "x", "y", "xtrain", "ytrain", "xtest" or "ytest"')
        if drop:
            for dr in self.get_drop(xy):
                df = df.drop(columns=dr)
        if not get_id:
            return df
        return self._makeid(df)

    def get_image_data(self, xy: str) -> 'np.ndarray':
        """
        Get image data.

        :param xy: Which dataframe
        :return: Image matrix
        """
        assert self._image_col != '', 'Image does not exist on current dataframe'
        assert self._is_split_train_test, _TRAIN_TEST_ERROR
        xy = xy.lower()
        if xy not in ['x', 'y', 'xtest', 'ytest', 'xtrain', 'ytrain']:
            raise ValueError('xy param must be "x", "y", "xtrain", "ytrain", "xtest" or "ytest"')
        xy += '_image'
        return self._train_test[xy]

    def find_id_in_df(self, column_id: str, obj_id: int) -> List[str]:
        """
        Returns the name of the dataframe for an object ID.

        :param column_id: Column ID
        :param obj_id: ID value
        :return: Dataframe name
        """
        assert self._is_split_train_test, _TRAIN_TEST_ERROR
        found_at = []
        for xy in ['x', 'y', 'xtrain', 'ytrain', 'xtest', 'ytest']:
            df: 'pd.DataFrame' = self.get_dataframe(xy=xy, get_id=True)
            if column_id not in df.columns:
                continue
            if len(df[df[column_id] == obj_id]) >= 1:
                found_at.append(xy)
        return found_at

    def get_metrics(self) -> 'pd.DataFrame':
        """
        :return: Returns metrics dataframe copy
        """
        return self._metrics.copy()

    def get_image_balance(self) -> Dict[str, Tuple[float, float]]:
        """
        :return: Returns image class balance (white/blacks)
        """
        assert self._image_col != '', 'Image does not exist on current dataframe'

        def _get_white_blacks(img_data: 'np.ndarray') -> Tuple[float, float]:
            """
            Returns white blacks percentages.

            :param img_data: Images
            :return: (white,black) percentages
            """
            w, _ = np.shape(img_data[0])
            total = len(img_data) * w * w
            white = np.sum(img_data)
            black = total - white
            # print(white, black, white / black, white / total, black / total)
            return white / total, black / total

        def _average_xy(img_data_x: 'np.ndarray', img_data_y: 'np.ndarray') -> Tuple[float, float]:
            """
            Compute average between two sets.

            :param img_data_x: Images in x
            :param img_data_y: Images in y
            :return: (white,black) percentages
            """
            xw, xb = _get_white_blacks(img_data_x)
            yw, yb = _get_white_blacks(img_data_y)
            return (xw + yw) / 2, (xb + yb) / 2

        def _get_df(df_name: str) -> Tuple[float, float]:
            """
            Compute white blacks by dataset.

            :param df_name: Data name
            :return: (white,black) percentages
            """
            x = self.get_image_data('x' + df_name)
            y = self.get_image_data('y' + df_name)
            return _average_xy(x, y)

        return {
            'dataset': _get_df(''),
            'train': _get_df('train'),
            'test': _get_df('test')
        }

    def _makeid(self, target: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Create a dataframe from ID columns.

        :param target: Target dataframe
        :return: ID dataframes
        """
        _id_data = {}
        for _col in self._id:
            _id_data[_col] = self.inverse_minmax_scaler(column_name=_col, x=target[_col].values)
        _df = pd.DataFrame(data=_id_data)
        _df.columns = self._id
        _df = _df.astype('int32')
        return _df

    def set_drop(self, *args: Union[List[str], str], **kwargs) -> None:
        """
        Set droppable columns.

        :param args: Drop columns ID
        :param kwargs: Optional keyword arguments
        """
        axis: str = str(kwargs.get('axis', 'xy')).lower()
        if axis not in ['x', 'y', 'xy']:
            raise ValueError(f'Invalid axis <{axis}>')
        c: str
        for c in args:
            if self.has_column(c, xy=axis):
                if axis == 'xy' or axis == 'x':
                    self._drop_cols['x'].append(c)
                if axis == 'xy' or axis == 'y':
                    self._drop_cols['y'].append(c)
            else:
                raise ValueError(f'Column <{c}> does not exist')

    def get_drop(self, xy: str) -> List[str]:
        """
        Returns droppable columns.

        :param xy: Which dataframe
        :return: Columns
        """
        xy = xy.lower()
        if xy in ['x', 'xdata', 'xtest', 'xtrain']:
            df = 'x'
        elif xy in ['y', 'ydata', 'ytest', 'ytrain']:
            df = 'y'
        else:
            raise ValueError(f'Invalid xy <{xy}>')
        return self._drop_cols[df].copy()

    def checkpoint(self) -> None:
        """
        Checkpoint data.
        """
        self._data_x_checkpoint = self._data_x.copy()
        self._data_y_checkpoint = self._data_y.copy()

    def restore(self) -> None:
        """
        Restore from last checkpoint.
        """
        self._data_x = self._data_x_checkpoint.copy()
        self._data_y = self._data_y_checkpoint.copy()

    def del_column(self, column_name: Union[str, List[str]], xy: str) -> None:
        """
        Delete column from data.

        :param column_name: Column name to delete
        :param xy: Which dataframe to delete from
        """
        if not isinstance(column_name, list):
            column_name = [column_name]
        colname: str
        for colname in column_name:
            if colname == '':
                continue
            if self._image_col != '':
                assert colname != self._image_col, 'Image column cannot be dropped'
            if self.has_column(column_name=colname, xy=xy):
                if xy == 'x' or xy == 'xy':
                    self._data_x = self._data_x.drop(columns=[colname])
                if xy == 'y' and xy == 'xy':
                    self._data_y = self._data_y.drop(columns=[colname])
            else:
                raise ValueError(f'Column <{colname}> does not exist in x dataframe')

    def _assert_shape(self) -> None:
        """
        Validate dataframe shape, x and y must be the same.
        """
        _shape_x = self._data_x.shape
        _shape_y = self._data_y.shape
        _msg = 'Datasets X and Y must have the same shape, but x shape {0} differs from y shape {1}'
        assert _shape_x[0] == _shape_y[0] and _shape_x[1] == _shape_y[1], \
            _msg.format(_shape_x, _shape_y)
        _col_x = self._data_x.columns
        _col_y = self._data_y.columns
        for i in range(len(_col_x)):
            assert _col_x[i] == _col_y[i], 'Datasets must have same column names'

    def reset(self, xy: str = 'xy') -> None:
        """
        Reset data.

        :param xy: Which dataframe
        """
        xy = xy.lower()
        if xy not in _VALID_COLUMN:
            raise ValueError(_XY_ERROR)
        if xy in _VALID_XY:
            self._reset_all()
            return
        if xy == 'x':
            self._data_x = self._raw_data_x.copy()
        elif xy == 'y':
            self._data_y = self._raw_data_y.copy()

    def has_column(self, column_name: str, xy: str = 'xy') -> bool:
        """
        Returns true if data has the given column. Both columns have the same names.

        :param column_name: Column name
        :param xy: Which dataframe
        :return: True if exists
        """
        if xy == 'x':
            return column_name in self._data_x.columns
        elif xy == 'y':
            return column_name in self._data_y.columns
        elif xy == 'xy':
            return column_name in self._data_x.columns and column_name in self._data_y.columns
        else:
            raise ValueError(_XY_ERROR)

    def add_single_column(self, xy: str, column_name: str, column_values: Union['np.ndarray', 'pd.Series']) -> None:
        """
        Add column to dataframe.

        :param xy: Which dataframe to add from
        :param column_name: New column name
        :param column_values: Column values
        """
        if self.has_column(column_name=column_name):
            raise ValueError(f'Column <{column_name}> already exists in data')
        assert xy in ['x', 'y', 'xy'], _VALID_XY
        if self._minmax_scaler_applied:
            raise RuntimeError('Cannot add column if minmax is applied')
        self._data_x[column_name] = column_values
        self._data_y[column_name] = column_values
        if xy == 'x':
            self.set_drop(column_name, axis='y')
        elif xy == 'y':
            self.set_drop(column_name, axis='x')

    def add_double_column(
            self,
            column_name: str,
            column_value_x: 'np.ndarray',
            column_value_y: 'np.ndarray'
    ) -> None:
        """
        Add double column to dataframe.

        :param column_name: Which dataframe to add from
        :param column_value_x: Value in x col
        :param column_value_y: Value in y col
        """
        if self.has_column(column_name=column_name):
            raise ValueError(f'Column <{column_name}> already exists')
        if self._minmax_scaler_applied:
            raise RuntimeError('Cannot add column if minmax is applied')
        self._data_x[column_name] = column_value_x
        self._data_y[column_name] = column_value_y

    def get_literal(self, column_name: str, value: int) -> str:
        """
        Returns the literal value from an integer position.

        :param column_name: Column name from the literals
        :param value: Value position
        :return: Literal name
        """
        if column_name not in self._literals.keys():
            raise ValueError(f'{column_name} does not exists in literals')
        assert 0 <= value < len(self._literals[column_name]), 'Value index overflows'
        return self._literals[column_name][value]

    def get_classification_literals(self) -> Dict[str, List[str]]:
        """
        :return: Returns classification literals library
        """
        return copy.deepcopy(self._literals)

    def _reset_all(self) -> None:
        """
        Reset all data.
        """
        self.reset(xy='x')
        self.reset(xy='y')
        self._minmax_scaler = preprocessing.MinMaxScaler()  # Create scale
        self._minmax_scaler_applied = False
        self.remove_train_test()

    def get_shape(self) -> Tuple[int, int]:
        """
        :return: Returns data shape (rows, columns)
        """
        # self._assert_shape()
        return self._data_x.shape[0], self._data_x.shape[1]

    def is_scaled(self) -> bool:
        """
        :return: Is data scaled
        """
        return self._minmax_scaler_applied

    def scale_image(self, to: Tuple[Union[int, float], Union[int, float]], dtype: Optional[str]) -> None:
        """
        Scale images to range.

        :param to: Range to scale
        :param dtype: Cast to data type
        """
        if dtype is None:
            print(f'Scaling images to range {to}:', end='')
        else:
            print(f'Scaling images to range {to} type {dtype}:', end='')
        print(' X...', end='')
        self._images_x = scale_array_to_range(self._images_x, to, dtype)
        print('OK, Y...', end='')
        self._images_y = scale_array_to_range(self._images_y, to, dtype)
        print('OK')

    def scale_minmax(self) -> None:
        """
        Apply minmax scaler to all data.
        """
        self._assert_shape()

        if self._minmax_scaler_applied:
            raise RuntimeError('Scaler already applied')
        self._minmax_scaler_applied = True

        x_values = self._data_x.values
        y_values = self._data_y.values

        for c in range(len(self._data_x.columns)):
            assert self._data_x.columns[c] == self._data_y.columns[c], \
                f'Column at pos <{c}> must be the same in x and y dataframes'

        # Scale min/max fitting
        # self._minmax_scaler.fit(np.vstack((x_values, y_values)))
        self._minmax_scaler.partial_fit(x_values)
        self._minmax_scaler.partial_fit(y_values)

        # Create new normalized data frames
        x_norm_pd = pd.DataFrame(self._minmax_scaler.transform(x_values))
        x_norm_pd.columns = self._data_x.columns

        y_norm_pd = pd.DataFrame(self._minmax_scaler.transform(y_values))
        y_norm_pd.columns = self._data_y.columns

        # Replace original ID columns values
        x_norm_pd[self._id] = self._data_x[self._id]
        y_norm_pd[self._id] = self._data_y[self._id]

        # Save
        self._data_x: 'pd.DataFrame' = x_norm_pd
        self._data_y: 'pd.DataFrame' = y_norm_pd
        self._scaler_col = []
        for _s in self._data_x.columns:
            self._scaler_col.append(_s)

    def describe(self, xy: str, get_id: bool = False) -> object:
        """
        Describe set.

        :param xy: Which dataframe
        :param get_id: Use ID dataframe
        :return: Object describer
        """
        return self.get_dataframe(xy, get_id=get_id).describe()

    def get_column_by_name(self, xy: str, column_name: str) -> 'pd.DataFrame':
        """
        Returns column values by name.

        :param xy: Which dataframe
        :param column_name: Column name
        :return: Column dataframe
        """
        self._assert_shape()
        xy = xy.lower()
        assert self.has_column(column_name=column_name), \
            f'Column <{column_name}> does not exist in dataframe "{xy}"'
        return self.get_dataframe(xy, get_id=column_name in self._id)[[column_name]]

    def _inverse_minmax_scaler(self, column_name: str, x: 'np.ndarray') -> 'np.ndarray':
        """
        Inverse from minmax scaler.

        :param column_name: Source column number from scaler
        :param x: array-like of shape (n_samples, n_features)
        """
        if column_name in self._id:
            return x  # The same

        if not self._minmax_scaler_applied:
            # warnings.warn('Warning, minmax has not been applied')
            return x

        # Create new scaler and update
        scaler = preprocessing.MinMaxScaler()
        scaler.min_, scaler.scale_ = self.get_minmax_scaler_limits(column_name)

        # Returns scaled data
        return scaler.inverse_transform(x)

    def get_minmax_scaler_limits(self, column_name: str) -> Tuple[float, float]:
        """
        Returns minmax scaler limits.

        :param column_name:
        :return: Limit tuple (min, scale)
        """
        if not self._minmax_scaler_applied:
            return 0, 0
        if column_name not in self._scaler_col:
            raise ValueError(f'Column <{column_name}> does not exist in scaler columns train data')
        column_name = self._scaler_col.index(column_name)
        return self._minmax_scaler.min_[column_name], self._minmax_scaler.scale_[column_name]

    def inverse_minmax_scaler(self, column_name: str, x: 'np.ndarray', force_int: bool = False) -> 'np.ndarray':
        """
        Inverse row vector from scaler.

        :param column_name: Column name
        :param x: Data column to inverse, shape (x, 1)
        :param force_int: Apply int() to each output element
        :return: Inversed data vector (x, 1)
        """
        assert len(np.shape(x)) == 1 or np.shape(x)[1] == 1, 'x must be a column, shape (x,1)'
        inversed: 'np.ndarray' = self._inverse_minmax_scaler(
            column_name=column_name, x=x.reshape(-1, 1)).reshape(1, -1)[0, :]
        if force_int:
            inversed = inversed.astype(int)
        return inversed

    def get_inversed_column_minmax_scaler(self, xy: str, column_name: str,
                                          as_vector: bool = True) -> Union['np.ndarray', 'pd.DataFrame']:
        """
        Get a column, inversed, by the column name identifier.

        :param xy: Which data, 'x' or 'y'
        :param column_name: Column name
        :param as_vector: If true, return vector, else, return dataframe object
        :return: Inversed data vector (1, x) or dataframe
        """
        col = self.inverse_minmax_scaler(
            column_name=column_name,
            x=self.get_column_by_name(xy=xy, column_name=column_name).values
        )
        if as_vector:
            return col
        else:
            p = pd.DataFrame(col)
            p.columns = [column_name]
            return p

    def remove_train_test(self) -> None:
        """
        Clear train test data.
        """
        try:
            del self._train_test
        except AttributeError:
            pass
        self._is_split_train_test = False
        self._split_train_test_mode = 'NONE'
        gc.collect()

    def save_train_test(
            self,
            test_size: float = 0.3,
            split_by_id: bool = False,
            split_by_id_column: str = '',
            split_by_session_id: Optional[Dict[str, Union['np.ndarray', 'pd.Series']]] = None
    ) -> None:
        """
        Save train test split.

        :param test_size: Test size in percentage
        :param split_by_id: Split using ID
        :param split_by_id_column: Name of the column ID
        :param split_by_session_id: Dict containing numpy ID split arrays (train_id: ..., test_id: ...)
        """
        self.remove_train_test()
        self._train_test = self.make_train_test(
            test_size=test_size,
            split_by_id=split_by_id,
            split_by_id_column=split_by_id_column,
            split_by_session_id=split_by_session_id
        )
        self._is_split_train_test = True

    def project_split_train_test(self, project_id_list: List[int], id_column: str) -> None:
        """
        Split train test.

        :param project_id_list: Project ID list
        :param id_column: Name of the column ID to split
        """
        if self._is_split_train_test:
            raise RuntimeError('Train test already split')

        # Make sure all given ids exists
        assert len(project_id_list) > 0, 'Object must be list type and not empty'
        assert self.has_column(id_column, xy='x'), 'ID column does not exist'
        id_col: 'pd.Series' = self._data_x[id_column]
        true_id = list(np.unique(id_col.values))
        for i in project_id_list:
            assert i in true_id, f'ID element <{i}> does not exist in database'
        project_id_list = list(np.unique(project_id_list))

        # Split data using ID list, 1. make mask
        mask_id = id_col == -1
        for i in project_id_list:
            mask_id = mask_id | (id_col == i)

        # Use mask to create test and train
        x_train = self._data_x[~mask_id].values
        x_test = self._data_x[mask_id].values
        y_train = self._data_y[~mask_id].values
        y_test = self._data_y[mask_id].values

        self.remove_train_test()
        self._train_test = self._make_train_test(x_train, x_test, y_train, y_test)
        self._is_split_train_test = True

    def make_train_test(
            self,
            test_size: float = 0.3,
            split_by_id: bool = False,
            split_by_id_column: str = '',
            split_by_session_id: Optional[Dict[str, Union['np.ndarray', 'pd.Series']]] = None
    ) -> Dict[str, Union['pd.DataFrame', 'np.ndarray']]:
        """
        Generate train test data.

        :param test_size: Test size in percentage
        :param split_by_id: Split using ID
        :param split_by_id_column: Name of the column ID
        :param split_by_session_id: Dict containing numpy ID split arrays
        :return: Train and test vectors
        """
        x_pd: 'pd.DataFrame' = self._data_x
        y_pd: 'pd.DataFrame' = self._data_y

        if split_by_id:
            assert split_by_id_column != '', 'ID column must be set'
            assert split_by_id_column in self._id, 'ID column not in data ID cols defined by the constructor'

        if split_by_session_id is None:
            assert 1 > test_size > 0, 'Test size must be between 0 and 1'
            if not split_by_id:
                x_train, x_test, y_train, y_test = train_test_split(x_pd.values, y_pd.values, test_size=test_size)
                self._split_train_test_mode = 'ALL'
            else:
                assert self.has_column(split_by_id_column, xy='x'), 'ID column does not exist'
                x_train, x_test, y_train, y_test = _train_test_split_project_id(x_pd, y_pd, test_size=test_size,
                                                                                id_column=split_by_id_column)
                self._split_train_test_mode = 'BY_ID'
        else:
            _x: 'np.ndarray' = x_pd.values
            _y: 'np.ndarray' = y_pd.values
            x_train = _x[split_by_session_id['train_id']]
            x_test = _x[split_by_session_id['test_id']]
            y_train = _y[split_by_session_id['train_id']]
            y_test = _y[split_by_session_id['test_id']]
            # self._split_train_test_mode will be stored by load_session

        return self._make_train_test(x_train, x_test, y_train, y_test)

    def _make_train_test(
            self,
            x_train: 'np.ndarray',
            x_test: 'np.ndarray',
            y_train: 'np.ndarray',
            y_test: 'np.ndarray'
    ) -> Dict[str, Union['pd.DataFrame', 'np.ndarray']]:
        """
        Make train test and save to dataset.

        :param x_train: Train values x
        :param x_test: Test values x
        :param y_train: Train values y
        :param y_test: Test values y
        :return: Train and test vectors
        """
        x_pd: 'pd.DataFrame' = self._data_x.copy()
        y_pd: 'pd.DataFrame' = self._data_y.copy()

        # Convert to data frame
        x_train_pd = pd.DataFrame(x_train)
        x_train_pd.columns = x_pd.columns
        x_test_pd = pd.DataFrame(x_test)
        x_test_pd.columns = x_pd.columns

        y_train_pd = pd.DataFrame(y_train)
        y_train_pd.columns = y_pd.columns
        y_test_pd = pd.DataFrame(y_test)
        y_test_pd.columns = y_pd.columns

        def get_image(df: 'pd.DataFrame', xy: str) -> 'np.ndarray':
            """
            Returns image numpy array from ID.

            :param df: Dataframe to use
            :param xy: Use "x" or "y"
            :return: Image array
            """
            if self._image_col == '':
                return np.ndarray((0, 0))
            im_id = self.inverse_minmax_scaler(self._image_col, df[[self._image_col]].values, force_int=True)
            if xy == 'x':
                return self._images_x[im_id]
            elif xy == 'y':
                return self._images_y[im_id]
            else:
                raise ValueError('Invalid xy, use "x" or "y"')

        # Generate output
        out = {

            # Split dataframes
            'xtrain': x_train_pd,
            'xtest': x_test_pd,
            'ytrain': y_train_pd,
            'ytest': y_test_pd,

            'x': x_pd,
            'y': y_pd,

            # Images
            'xtrain_image': get_image(x_train_pd, 'x'),
            'xtest_image': get_image(x_test_pd, 'x'),
            'ytrain_image': get_image(y_train_pd, 'y'),
            'ytest_image': get_image(y_test_pd, 'y'),

            'x_image': get_image(x_pd, 'x'),
            'y_image': get_image(y_pd, 'y')

        }

        if self._image_col != '':
            assert len(out['x']) == len(out['x_image']), 'X number of elements must be the same'
            assert len(out['xtrain']) == len(out['xtrain_image']), 'X train number of elements must be the same'
            assert len(out['xtest']) == len(out['xtest_image']), 'X test number of elements must be the same'

            assert len(out['y']) == len(out['y_image']), 'Y number of elements must be the same'
            assert len(out['ytrain']) == len(out['ytrain_image']), 'Y train number of elements must be the same'
            assert len(out['ytest']) == len(out['ytest_image']), 'Y test number of elements must be the same'

        # Returns data
        return out
