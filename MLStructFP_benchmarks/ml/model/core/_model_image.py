"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - CORE - MODEL IMAGE

Model based on xy images.
"""

__all__ = ['GenericModelImage']

from abc import ABCMeta

from MLStructFP_benchmarks.ml.model.core import GenericModel, ModelDataXY
from MLStructFP_benchmarks.ml.model.core._model import _PATH_SESSION, _ERROR_MODEL_IN_PRODUCTION

from pandas.util import hash_pandas_object
from typing import Tuple, Dict, Any, Callable, List, Optional
import math
import numpy as np
import os
import pandas as pd


def _assert_shape(x: 'np.ndarray', y: 'np.ndarray', msg: str) -> None:
    """
    Assert shapes.

    :param x: X shape (images)
    :param y: Y shape (images)
    :param msg: Error message
    """
    xx = x.shape
    yy = y.shape
    assert xx == yy, 'Shape does not match at ' + msg


# noinspection PyUnusedLocal
class GenericModelImage(GenericModel, metaclass=ABCMeta):
    """
    Image model.
    """
    _assert_data: bool
    _data: 'ModelDataXY'
    _img_size: int
    _img_channels: int
    _x_img: 'np.ndarray'
    _x_train_img: 'np.ndarray'
    _x_test_img: 'np.ndarray'
    _y_img: 'np.ndarray'
    _y_train_img: 'np.ndarray'
    _y_test_img: 'np.ndarray'

    def __init__(self, data: Optional['ModelDataXY'], name: str, *args, **kwargs) -> None:
        """
        Constructor model images.

        :param data: Model data
        :param name: Model name
        :param img_size: Image size (px)
        :param img_channels: Number of channels of the image
        :param args: Optional non-keyword arguments
        :param kwargs: Optional keyword arguments
        """
        img_size: Optional[int] = kwargs.get('img_size', None)
        img_channels: Optional[int] = kwargs.get('img_channels', None)
        GenericModel.__init__(self, name=name, path=kwargs.get('path', ''))

        if data is None:
            assert isinstance(img_size, int), 'Image size must be defined'
            assert isinstance(img_channels, int), 'Image number of channels must be defined'
            assert img_size > 0, 'Image size cannot be zero'
            assert img_channels >= 1, 'Number of channels must equal or greater than 1'
            assert math.log(img_size, 2).is_integer(), 'Image size must be a power of 2'
            self._img_size = img_size
            self._img_channels = img_channels
            self._assert_data = False
            return

        # Save data
        assert data.__class__.__name__ == 'ModelDataXY', \
            f'Invalid data class <{data.__class__.__name__}>'
        self._data = data
        self._assert_data = True

        # Get images
        self._x_img = self._data.get_image_data('x')
        self._x_train_img = self._data.get_image_data('xtrain')
        self._x_test_img = self._data.get_image_data('xtest')

        self._y_img = self._data.get_image_data('y')
        self._y_train_img = self._data.get_image_data('ytrain')
        self._y_test_img = self._data.get_image_data('ytest')

        # Assert shapes
        _assert_shape(self._x_train_img, self._y_train_img, 'train image data')
        _assert_shape(self._x_test_img, self._y_test_img, 'test image data')
        _assert_shape(self._x_img, self._y_img, 'image data')

        self._test_split = self._x_test_img.shape[0] / (self._x_test_img.shape[0] + self._x_train_img.shape[0])

        # Get image size
        shape = np.shape(self._x_img[0])
        if len(shape) == 2:
            self._img_channels = 1
        elif len(shape) == 3:
            self._img_channels = shape[2]
        else:
            raise ValueError('Invalid number of dimensions of the image')
        img_width = shape[0]
        img_height = shape[1]
        assert img_width == img_height, 'Image size must be the same in width and height, current {0}x{1}'.format(
            img_width, img_height)
        if img_size is not None:
            assert img_size == img_width, \
                'Image size from loaded data is different than the value provided to the constructor'
        if img_channels is not None:
            assert img_channels == self._img_channels, \
                'Image channels from loaded data are different than the value provided to the constructor'
        self._img_size = img_width

        # Reshape
        self._x_img = self.reshape_image(self._x_img)
        self._x_train_img = self.reshape_image(self._x_train_img)
        self._x_test_img = self.reshape_image(self._x_test_img)
        self._y_img = self.reshape_image(self._y_img)
        self._y_train_img = self.reshape_image(self._y_train_img)
        self._y_test_img = self.reshape_image(self._y_test_img)

    def reshape_image(self, img: 'np.ndarray') -> 'np.ndarray':
        """
        Reshape image for current model.

        :param img: Image to reshape
        :return: Reshaped image
        """
        if len(img.shape) == 3:
            return img.reshape((-1, self._img_size, self._img_size, self._img_channels))
        else:
            return img

    def unreshape_image(self, img: 'np.ndarray') -> 'np.ndarray':
        """
        Unreshape image for current model.

        :param img: Image to unreshape
        :return: Unreshaped image
        """
        assert len(img.shape) == 4, 'Only 4 dimensional arrays are allowed'
        return img.reshape((len(img), self._img_size, self._img_size))

    def get_image_size(self) -> int:
        """
        :return: Returns the image size (width/height) in px
        """
        return self._img_size

    def get_image_num_channels(self) -> int:
        """
        :return: Returns the number of channels of the image
        """
        return self._img_channels

    def get_images(self, xy: str) -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Returns the xy images.

        :param xy: Which data
        :return: Image arrays
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if xy == 'dataset':
            return self._x_img, self._y_img
        elif xy == 'train':
            return self._x_train_img, self._y_train_img
        elif xy == 'test':
            return self._x_test_img, self._y_test_img
        else:
            raise ValueError('Invalid xy parameter, valid "dataset", "train" or "test"')

    def _get_accuracy_df(
            self,
            xy: str,
            item_id_col: str,
            notebook_tqdm: bool,
            fun: Callable[[int], List[float]],
            use_model: bool
    ) -> 'pd.DataFrame':
        """
        Compute model accuracy.

        :param xy: Dataframe to test
        :param item_id_col: Unique ID column name
        :param notebook_tqdm: Use TQDM
        :param fun: Function that updates each accuracy index
        :param use_model: Use model
        :return: Accuracy values
        """

        def _print_stats(dataf: 'pd.DataFrame') -> None:
            """
            Print stats.

            :param dataf: Loaded data
            """
            fround = 4
            print('Mean:\t{0}\nStd:\t{1}'.format(round(float(np.mean(dataf['accuracy'])), fround),
                                                 round(float(np.std(dataf['accuracy'])), fround)))

        acc_key: str = 'accuracy-df-' + xy
        if not use_model:
            acc_key += '-no-model'
        if self._exists_in_train_data(acc_key):
            print(f'Reading from stored file: {self._get_train_data(acc_key)}')
            rdf = pd.read_csv(self._get_train_data(acc_key))
            _print_stats(rdf)
            return rdf

        if not notebook_tqdm:
            from tqdm import tqdm
        else:
            from tqdm.notebook import tqdm

        if xy not in ['test', 'train', 'dataset']:
            raise ValueError('Invalid xy parameter, valid "dataset", "train" or "test"')
        if xy == 'dataset':
            xy = ''
        id_col = self._data.get_dataframe(xy='x' + xy, get_id=True)
        acc_pd = pd.DataFrame(id_col[item_id_col])
        acc_pd['accuracy'] = 0

        acc: float = -1
        j = 0
        for k in self.get_metric_names():
            if 'val_' in k:
                continue
            if 'accuracy' in k:
                acc = j
                break
            j += 1
        if acc == -1:
            raise RuntimeError('Accuracy not found at metrics, check out if accuracy on loss')

        o_verbose = self._verbose
        self._verbose = False
        finished = False
        try:
            for i in tqdm(range(len(acc_pd))):
                acc_pd.iloc[i, 1] = fun(i)[j]
            finished = True
        except KeyboardInterrupt:
            print('Interrupted by user')
        self._verbose = o_verbose

        # Save accuracy pandas file
        if finished:
            acc_file: str = _PATH_SESSION + os.path.sep + \
                            f'train_accuracy_df_{abs(int(hash_pandas_object(acc_pd).sum()))}.csv'
            acc_pd.to_csv(acc_file, index=False)
            print(f'Saving to file: {acc_file}')
            self._register_train_data(acc_key, acc_file)

        _print_stats(acc_pd)
        return acc_pd

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        See upper doc.
        """
        data['hash_images'] = self._data.get_data_hash('images')

        # Data info
        data['data_filename'] = self._data.get_filename()
        data['data_image_sz'] = self.get_image_size()
        data['data_image_num_channels'] = self.get_image_num_channels()

    def _custom_load_session(
            self,
            filename: str,
            asserts: bool,
            data: Dict[str, Any],
            check_hash: bool
    ) -> None:
        """
        See upper doc.
        """
        if asserts and check_hash:
            if self._assert_data:
                assert data['hash_images'] == self._data.get_data_hash('images'), 'Data images hash changed'
            assert data['data_image_sz'] == self.get_image_size()
            assert data['data_image_num_channels'] == self.get_image_num_channels()
