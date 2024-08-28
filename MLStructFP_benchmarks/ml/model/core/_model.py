"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - CORE - MODEL

Generic model class.
"""

__all__ = [
    'GenericModel',
    '_ERROR_MODEL_IN_PRODUCTION',
    '_ERROR_MODEL_NOT_IN_PRODUCTION',
    '_PATH_SESSION',
    '_SESSION_EXPORT_VERSION'
]

from MLStructFP_benchmarks.ml.utils import file_md5, load_history_from_csv
from MLStructFP_benchmarks.ml.utils.callbacks import TensorBoardv2, TimeHistory
from MLStructFP_benchmarks.ml.utils.keras_engine import fit_loop

from keras.backend import is_tensor
from keras.backend.tensorflow_backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.callbacks import History
from keras.engine import training_utils
from keras.layers import Layer
from keras.models import model_from_json, Model
from keras.optimizers import Optimizer
from keras.utils.generic_utils import slice_arrays
from keras.utils.layer_utils import count_params
from keras_tqdm import TQDMNotebookCallback  # https://github.com/bstriner/keras-tqdm
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Callable, Iterator, Optional
import copy
import datetime
import difflib
import gc
import hashlib
import json
import numpy as np
import os
import pandas as pd
import re
import shutil
import tensorflow as tf
import time
import traceback
import warnings

os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

_ERROR_MODEL_COMPILED: str = 'Model has already been compiled'
_ERROR_MODEL_IN_PRODUCTION: str = 'Model is in production mode'
_ERROR_MODEL_NOT_COMPILED: str = 'Model has not been compiled yet, use .compile(*args)'
_ERROR_MODEL_NOT_IN_PRODUCTION: str = 'Model is not in production mode'
_ERROR_MODEL_NOT_TRAINED: str = 'Model has not been trained yet, use .train(*args)'
_ERROR_MODEL_TRAINED: str = 'Model has already been trained, it cannot be modified'

_RUNTIME_METADATA_KEY: str = 'run_data'
_SESSION_EXPORT_VERSION: str = '2.3'

_TRAIN_REDIRECT_METRICS: str = 'metrics'
_TRAIN_REDIRECT_NONE: str = ''

# Path definition
_PATH_CHECKPOINT: str = '.checkpoint'
_PATH_LOGS: str = '.logs'
_PATH_SESSION: str = '.session'


def _hash_list(a: List[Any]) -> str:
    """
    Returns list hash.

    :param a: List object
    :return: Hash
    """
    h5 = hashlib.md5()
    h5.update(str(len(a)).encode())
    for x in a:
        h5.update(str(x).encode())
    return h5.hexdigest()


def _check_path(path: str) -> None:
    """
    Check paths and generate them if not exist.
    """
    path_logs = os.path.join(path, _PATH_LOGS)
    path_cp = os.path.join(path, _PATH_CHECKPOINT)
    path_session = os.path.join(path, _PATH_SESSION)
    if not os.path.isdir(path_logs):
        e = f'Logs path <{path_logs}> does not exist. Creating new one'
        warnings.warn(e)
        os.mkdir(path_logs)
    if not os.path.isdir(path_cp):
        e = f'Checkpoint path <{path_cp}> does not exist. Creating new one'
        warnings.warn(e)
        os.mkdir(path_cp)
    if not os.path.isdir(path_session):
        e = f'Session path <{path_session}> does not exist. Creating new one'
        warnings.warn(e)
        os.mkdir(path_session)


def _reset_weight(model: 'Model') -> None:
    """
    Reset model weight.

    :param model: Model
    """
    layer: Union['Layer', 'Model']
    for layer in model.layers:
        if isinstance(layer, Model):  # If you're using a model as a layer
            _reset_weight(layer)  # Apply function recursively
            continue

        # Where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if 'initializer' not in key:  # is this item an initializer?
                continue  # if not, skip it

            # Find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer':  # special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace('_initializer', ''))

            # Use the initializer
            if var is None:
                continue
            var.assign(initializer(var.shape, var.dtype))


def _normalize_arch_json(arch: str) -> str:
    """
    Make normalized architecture for comparision.

    :param arch: Arch JSON file
    :return: Normalized names JSON file
    """
    sm: List[str] = arch.split('\n')
    assert len(sm) > 1, 'Invalid JSON model architectures, use model.to_json(indent=2)'

    # Replace all names by stings
    namekeys = {}
    for i in sm:
        if '"name": "' in i:
            j = i.split(':')
            j.pop(0)
            j = ':'.join(j).replace('"', '').strip().replace(',', '')
            namekeys[j] = f'layer_{len(namekeys.keys())}'

    # Replace all names
    sm2: str = arch
    for k in namekeys.keys():
        sm2 = sm2.replace(f'"{k}"', f'"{namekeys[k]}"')

    sm: List[str] = sm2.split('\n')

    # Find functions and replace memory address
    for i in range(len(sm)):
        if '"function": [' in sm[i]:
            funcl: List[str] = sm[i + 1].split('\\n')
            if len(funcl) > 2:
                sm[i + 1] = funcl[0]

    return '\n'.join(sm)


class GenericModel(ABC):
    """
    Generic model class.
    """
    # Model
    _check_compilation: bool
    _compile_config: Dict[str, Union[str, float, Callable]]
    _custom_metrics: List[str]
    _custom_stateful_metrics: Optional[List[str]]
    _is_compiled: bool
    _loaded_session: Dict[str, str]
    _model: 'Model'
    _name: str
    _name_formatted: str
    _output_layers: List[str]
    _path: str
    _print_enabled: bool
    _session_data: Dict[str, Union[str, float, list, dict]]
    _verbose: bool
    _version: str

    # Train
    _history: Dict[str, List[float]]
    _is_trained: bool
    _metric_test: List[float]
    _metric_train: List[float]
    _test_split: float
    _train_metadata: Dict[str, Any]
    _train_redirect: str

    # Callbacks
    _early_stopping: Dict[str, Any]
    _model_checkpoint: Dict[str, Any]
    _reduce_lr_on_plateau: Dict[str, Any]
    _tqdm_leave_inner: bool
    _use_csv_logger: bool
    _use_fit_logger: bool
    _use_tensorboard: bool
    _use_tqdm_notebook: bool

    # Others
    _load_session_indent: int

    def __init__(self, name: str, path: Union[str, Path] = '') -> None:
        """
        Constructor.

        :param name: Model name
        :param path: Working path
        """
        ABC.__init__(self)
        if hasattr(self, '_name'):  # If true, the constructor has already been called
            return
        self._production = False  # Set model immutable to changes, also test and train data does not exist
        self._path = str(path)
        if self._path != '':
            self._path += os.path.sep
        _check_path(self._path)

        # Model
        self.set_name(name)
        self._check_compilation = True  # Check compilation during model
        self._compile_config = {}
        self._custom_metrics = []  # Metrics appended to default train metrics provided by the model
        self._custom_stateful_metrics = None  # If overridden, changes the default train metrics provided by the model
        self._is_compiled = False
        self._loaded_session = {}
        self._output_layers = []
        self._print_enabled = True
        self._session_data = {}
        self._verbose = False
        self._version = '1.0'

        # Training
        self._continue_train_count = 1
        self._history = {}
        self._is_trained = False
        self._metric_test = []
        self._metric_train = []
        self._test_split = 0
        self._train_metadata = {}
        self._train_redirect = _TRAIN_REDIRECT_NONE

        # Callbacks
        self._early_stopping = {'enabled': False}
        self._model_checkpoint = {'enabled': False}
        self._reduce_lr_on_plateau = {'enabled': False}
        self._tqdm_leave_inner = False  # Display validation info on run
        self._use_csv_logger = True
        self._use_fit_logger = True
        self._use_tensorboard = False
        self._use_tqdm_notebook = True

        # Others
        self._load_session_indent = 0

    def _print(self, msg) -> None:
        """
        Print message to console.

        :param msg: Message
        """
        if self._print_enabled:
            print('\t' * self._load_session_indent + msg)

    def get_path(self) -> str:
        """
        :return: Get path of the model files
        """
        return self._path

    def enable_production(self) -> None:
        """
        Set model in production.
        """
        if self._production:
            raise RuntimeError('Model already in production mode')
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        self._production = True

    def in_production(self) -> bool:
        """
        :return: Production mode status
        """
        return self._production

    def _register_session_data(self, key: str, val: Union[str, float, list, dict, tuple]) -> None:
        """
        Register data to session.
        Session data are invariant at train. Also, the data is extended from the loaded session data.

        :param key: Key
        :param val: Value
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if key in self._session_data.keys():
            raise KeyError(f'Key <{key}> already exists')
        if isinstance(val, (list, dict)):
            self._session_data[key] = val.copy()
        else:
            self._session_data[key] = val

    def _get_session_data(self, key: str) -> Optional[Union[str, float, list, dict, tuple]]:
        """
        Get session data from key. None if not exists.

        :param key: Key data
        :return: Value
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if key not in self._session_data.keys():
            return None
        val = self._session_data[key]
        if isinstance(val, (list, dict)):
            return val.copy()
        return val

    def _remove_session_data(self, key: str) -> None:
        """
        Remove data from session.

        :param key: Key data to remove
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if key not in self._session_data.keys():
            raise KeyError(f'Key <{key}> does not exists')
        del self._session_data[key]

    def set_name(self, name: str) -> None:
        """
        Set model name.

        :param name: Name
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self._name = name.strip().replace('-', '')
        alkey: str = 'MLAiSPACECHAR'
        self._name_formatted = self._name.lower().replace(' ', alkey)
        self._name_formatted = re.sub('[\W_]+', '', self._name_formatted).replace(alkey, '_')
        for _ in range(5):
            self._name_formatted = self._name_formatted.replace('__', '_')
        assert len(self._name_formatted) > 0, 'Invalid model name'

    def enable_verbose(self) -> None:
        """
        Enables model verbose.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self._verbose = True

    def enable_tensorboard(self) -> None:
        """
        Enable tensorboard logging.
        https://keras.io/api/callbacks/tensorboard/
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        self._use_tensorboard = True

    def enable_early_stopping(self, monitor: str = 'val_loss', patience: int = 50, mode: str = 'auto') -> None:
        """
        Enable early stopping.
        https://keras.io/api/callbacks/early_stopping/

        :param monitor: Monitor to test the overfitting
        :param patience: Patience epochs
        :param mode: Mode
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        assert patience > 1
        self._early_stopping['enabled'] = True
        self._early_stopping['mode'] = mode
        self._early_stopping['monitor'] = monitor
        self._early_stopping['patience'] = patience

    def enable_model_checkpoint(self, monitor: str = 'val_loss', epochs: int = 25, mode: str = 'auto') -> None:
        """
        Enable model checkpoint.
        https://keras.io/api/callbacks/model_checkpoint/

        :param monitor: Metric monitor
        :param epochs: Number of epochs to save the model
        :param mode: Mode
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        assert epochs >= 1
        self._model_checkpoint['enabled'] = True
        self._model_checkpoint['mode'] = mode
        self._model_checkpoint['monitor'] = monitor
        self._model_checkpoint['period'] = epochs

    def enable_reduce_lr_on_plateau(
            self,
            monitor: str = 'val_loss',
            factor: float = 0.1,
            patience: int = 25,
            mode: str = 'auto',
            min_delta: float = 0.0001,
            cooldown: int = 0,
            min_lr: float = 0
    ) -> None:
        """
        Enable reduce lr on plateau.
        https://keras.io/api/callbacks/reduce_lr_on_plateau/

        :param monitor: Quantity to be monitored.
        :param factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
        :param patience: Number of epochs with no improvement after which learning rate will be reduced
        :param mode: Mode
        :param min_delta: Min delta
        :param cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced
        :param min_lr: Lower bound on the learning rate
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        assert factor > 0
        assert patience > 1
        assert min_delta > 0
        self._reduce_lr_on_plateau['cooldown'] = cooldown
        self._reduce_lr_on_plateau['enabled'] = True
        self._reduce_lr_on_plateau['factor'] = factor
        self._reduce_lr_on_plateau['min_delta'] = min_delta
        self._reduce_lr_on_plateau['min_lr'] = min_lr
        self._reduce_lr_on_plateau['mode'] = mode
        self._reduce_lr_on_plateau['monitor'] = monitor
        self._reduce_lr_on_plateau['patience'] = patience

    def disable_tqdm(self, enable_logger_fit: bool = False) -> None:
        """
        Disables tqdm notebook.
        https://github.com/tqdm/tqdm

        :param enable_logger_fit: Enables fit logger
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        self._use_tqdm_notebook = False
        self._use_fit_logger = enable_logger_fit

    def disable_console_print(self) -> None:
        """
        Disables console printing.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self._print_enabled = False

    def disable_csv_logger(self) -> None:
        """
        Disables csv logger.
        https://keras.io/api/callbacks/csv_logger/
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)
        self._use_csv_logger = False

    def __cout__(self) -> str:
        _layers: str = ''
        if not self._production:
            if self._early_stopping['enabled']:
                _layers += f"(EARLY STOPPING:ON@{self._early_stopping['patience']})"
            else:
                _layers += '(EARLY STOPPING:OFF)'
            _layers += f' (TEST SPLIT {self._test_split * 100:.1f}%)'
        else:
            _layers = 'Production mode'
        return self._name + '\n' + _layers

    def __str__(self) -> str:
        return self.__cout__()

    def __repr__(self) -> str:
        return self.__cout__()

    def _load_history_from_csv(self, csv_logger_file: str) -> None:
        """
        Load history from file.

        :param csv_logger_file: Logger file
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        self._history = load_history_from_csv(csv_logger_file)
        self._train_metadata['epochs'] = len(self._history[list(self._history.keys())[0]])

    def _load_weights_from_file(self, weights_file: str) -> None:
        """
        Load weights from file.

        :param weights_file: Weights file
        """
        assert os.path.isfile(weights_file), f'Weights file <{weights_file}> does not exist'
        self.reset_weights(print_status=False)
        self._model.load_weights(weights_file)
        self._is_trained = True

    @abstractmethod
    def train(self, epochs: int, batch_size: int, val_split: float, shuffle: bool = True, **kwargs) -> None:
        """
        Fit model.

        :param epochs: Number of epochs
        :param batch_size: Number of the batch size
        :param val_split: Split for validation data (0.2 suggested)
        :param shuffle: Shuffle the data
        :param kwargs: Optional keyword arguments
        """
        pass

    def reset_train(self) -> None:
        """
        Reset train.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self._is_trained = False
        self.reset_weights()

        self._history = {}
        self._metric_test = []
        self._metric_train = []
        self._test_split = 0
        self._train_metadata = {}

    # noinspection PyMethodMayBeStatic
    def _format_tuple(
            self,
            x: Any,
            types: Optional[Union[Tuple, str]] = None,
            xy: str = ''
    ) -> Union['np.ndarray', List['np.ndarray']]:
        """
        Cast tuple to numpy array lists.

        :param x: Data vector
        :param types: Types vector, "np": numpy ndarray, "pd": pandas dataframe or series
        :param xy: Name of the xy vector
        :return: Numpy array vector
        """
        if isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
            x = [x]
        if types is not None and xy != '':
            if not isinstance(types, (tuple, list)):
                types = [types]
            assert len(x) == len(types), 'Types must have the same length as input vector'
            for j in range(len(x)):
                ty: str = types[j]
                if ty == 'np':
                    assert isinstance(x[j], np.ndarray), \
                        f'{xy}[{j}] must be a numpy ndarray'
                elif ty == 'pd':
                    assert isinstance(x[j], (pd.DataFrame, pd.Series)), \
                        f'{xy}[{j}] must be a pandas dataframe or series'
                else:
                    raise ValueError(f'Unrecognized type <{ty}> at types type pos <{j}>')
        y: List['np.ndarray'] = []
        j = 0
        for v in x:
            if isinstance(v, np.ndarray):
                y.append(v)
            elif isinstance(v, (pd.DataFrame, pd.Series)):
                y.append(v.values)
            else:
                raise ValueError(f'Unrecognized type <{type(v)}> at tuple pos <{j}>')
            j += 1
        if len(y) > 1:
            return y
        return y[0]

    # noinspection PyMethodMayBeStatic
    def _make_shape(
            self,
            x: Union[Tuple[Union['np.ndarray', 'pd.DataFrame', 'pd.Series']], 'np.ndarray', 'pd.DataFrame', 'pd.Series']
    ) -> Union[Tuple[Tuple], Tuple]:
        """
        Make shape from input vector.

        :param x: Input vector
        :return: Shape tuple
        """
        shp: List[Tuple] = []
        if not isinstance(x, (tuple, list)):
            x = [x]
        for v in x:
            assert isinstance(v, (np.ndarray, pd.DataFrame, pd.Series))
            shp.append(v.shape)
        if len(shp) > 1:
            return tuple(shp)
        else:
            return shp[0]

    # noinspection PyMethodMayBeStatic
    def _split_df_columns(
            self,
            df: Union['pd.DataFrame', 'pd.Series'],
            cols: List[str],
            as_list: bool = False
    ) -> Union[List, Tuple]:
        """
        Split dataframe using columns into a list or tuple of series.

        :param df: Dataframe
        :param cols: Use columns to split
        :param as_list: Returns list instead of tuple
        :return: List or tiple
        """
        vals: List['pd.Series'] = []
        for c in cols:
            vals.append(df[c])
        if as_list:
            return vals
        return tuple(vals)

    def get_memory_usage(self, batch_size: int) -> Tuple[float, int, int]:
        """
        Return the model memory usage in gb.

        :param batch_size: Batch size
        :return: (Memory VRAM used in gb, number of trainable weights, non-trainable weights)
        """
        return self._get_model_memory_usage(batch_size, self._model)

    def _get_model_memory_usage(self, batch_size: int, model: 'Model') -> Tuple[float, int, int]:
        """
        Get the model memory usage in gb.

        :param batch_size: Batch size
        :param model: Model
        :return: Memory VRAM used in gb, plus the number of parameters
        """
        try:
            from keras import backend as k
        except ImportError:
            from tensorflow.keras import backend as k

        shapes_mem_count = 0
        internal_model_mem_count = 0
        for lay in model.layers:
            layer_type = lay.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self._get_model_memory_usage(batch_size, lay)[0]
            single_layer_mem = 1
            out_shape = lay.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([k.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([k.count_params(p) for p in model.non_trainable_weights])
        # print(trainable_count, non_trainable_count)

        number_size = 4.0
        if k.floatx() == 'float16':
            number_size = 2.0
        if k.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = total_memory / (1024.0 ** 3) + internal_model_mem_count
        return gbytes, int(trainable_count), int(non_trainable_count)

    def _train(
            self,
            xtrain: Any,
            ytrain: Any,
            xtest: Any,
            ytest: Any,
            epochs: int,
            batch_size: int,
            val_split: float,
            shuffle: bool,
            use_custom_fit: bool,
            continue_train: bool,
            compute_metrics: bool
    ) -> None:
        """
        Fit model.

        :param xtrain: Metrics train data on x
        :param ytrain: Metrics train data true values, y=Ƒ(x)
        :param xtest: Metrics test data on x
        :param ytest: Metrics test data true values, y=Ƒ(x)
        :param epochs: Number of epochs
        :param batch_size: Number of the batch size
        :param val_split: Split for validation data (0.2 suggested)
        :param shuffle: Shuffle the data
        :param use_custom_fit: Use custom model fit on batch
        :param continue_train: Continue train from last train setup
        :param compute_metrics: Compute metrics after model train
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)

        # Redirect flow
        if self._train_redirect == _TRAIN_REDIRECT_METRICS:
            self._print('Redirecting train to metrics')
            return self._compute_metrics(xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest)
        elif self._train_redirect == _TRAIN_REDIRECT_NONE:
            pass
        else:
            raise RuntimeError(f'Invalid train redirect parameter <{self._train_redirect}>')

        # Asserts
        if not self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_NOT_COMPILED)
        if not continue_train:
            if self._is_trained:
                raise RuntimeError(_ERROR_MODEL_TRAINED)
        else:
            if not self._is_trained:
                raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        assert 0 < val_split < 1, 'Validation split must be a number between 0 and 1'
        assert epochs > 0, 'Number of epochs must be greater than zero'
        assert batch_size > 0, 'Batch size must be greater than zero'
        if not shuffle:
            e = 'Batch data shuffle is not enabled'
            warnings.warn(e)
        if compute_metrics:
            assert xtest is not None and ytest is not None, 'Test data cannot be None'

        tini = time.time()
        date_files: str = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        date_train: str = datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')

        # Remove status
        continue_train_count = 0
        prev_history: Dict[str, List[float]] = copy.deepcopy(self._history)
        if not continue_train:
            # self.reset_train()
            pass
        else:
            continue_train_count = self._train_metadata['continue_train_count'] + 1
            self._print(f'Continuing train, total: {continue_train_count}')

        # Create callbacks
        _callbacks = []

        # If early stopping is enabled
        if self._early_stopping['enabled']:
            _callbacks.append(
                EarlyStopping(
                    monitor=self._early_stopping['monitor'],
                    patience=self._early_stopping['patience'],
                    verbose=self._verbose and self._print_enabled,
                    mode=self._early_stopping['mode']
                )
            )
            self._print('[Callback] Using early stopping, '
                        'patience {0} mode {1}'.format(self._early_stopping['patience'],
                                                       self._early_stopping['mode']))

        # TQDM Notebooks
        if self._use_tqdm_notebook:
            if self._verbose:
                self._print('[Callback] TQDM cannot be used because verbose is active')
            else:
                self._print('[Callback] Using TQDM notebook')
                if self._tqdm_leave_inner:
                    self._print('[Callback] TQDM leaving inner data')
                _callbacks.append(TQDMNotebookCallback(
                    metric_format='{name}: {value:0.4f}',
                    leave_inner=self._tqdm_leave_inner
                ))

        # Tensorboard
        tensorboard_log: str = \
            '{0}{3}{1}{3}{2}'.format(os.path.join(self._path, _PATH_LOGS),
                                     self._name_formatted + '_' + datetime.datetime.today().strftime('%Y%m%d%H%M%S'),
                                     '',
                                     os.path.sep)
        if self._use_tensorboard:
            _callbacks.append(TensorBoardv2(log_dir=tensorboard_log))
            self._print(f'[Callback] Using tensorboard, log path: {tensorboard_log}')

        # Checkpoint on each epoch
        model_checkpoint_root = f'{_PATH_CHECKPOINT}{os.path.sep}{self._name_formatted}'
        model_checkpoint_path = model_checkpoint_root + '{2}{0}{2}{1}'.format(date_files, '', os.path.sep)
        if self._model_checkpoint['enabled']:
            _callbacks.append(ModelCheckpoint(
                filepath=os.path.join(model_checkpoint_path, 'epoch_{epoch:02d}.hdf5'),
                monitor=self._model_checkpoint['monitor'],
                verbose=self._verbose and self._print_enabled,
                save_weights_only=True,
                mode=self._model_checkpoint['mode'],
                period=self._model_checkpoint['period'])
            )
            self._print('[Callback] Using model checkpoint, path: {0} at {1} epochs mode {2}'.format(
                model_checkpoint_path, self._model_checkpoint['period'], self._model_checkpoint['mode']))

            # Create root folder if not exists
            if not os.path.isdir(model_checkpoint_root):
                os.mkdir(model_checkpoint_root)
            if not os.path.isdir(model_checkpoint_path):
                os.mkdir(model_checkpoint_path)

        def remove_checkpoint_empty() -> None:
            """
            Remove checkpoint folder if empty.
            """
            if not self._model_checkpoint['enabled']:
                return
            if 'epochs' not in list(self._train_metadata.keys()):
                self._train_metadata['epochs'] = 0
            if self._train_metadata['epochs'] < self._model_checkpoint['period']:
                self._print(f'Removing empty checkpoint path {model_checkpoint_path}')
                shutil.rmtree(model_checkpoint_path, ignore_errors=True)
            # If root folder is empty
            try:
                if len(os.listdir(model_checkpoint_root)) == 0:
                    shutil.rmtree(model_checkpoint_root, ignore_errors=True)
            except FileNotFoundError:
                pass

        # CSV Logger
        csv_logger_file: str = '{0}{3}{1}_{2}.csv'.format(os.path.join(self._path, _PATH_LOGS), self._name_formatted,
                                                          date_files, os.path.sep)
        if self._use_csv_logger and continue_train:
            csv_logger_file = self._train_metadata['csv_logger_file']
        csv_logger: 'CSVLogger' = CSVLogger(csv_logger_file)
        if self._use_csv_logger:
            _callbacks.append(csv_logger)
            self._print(f'[Callback] Using model CSV logger, file: {csv_logger_file}')

        def remove_csv_log() -> None:
            """
            Remove CSV log file.
            """
            if not self._use_csv_logger:
                return
            self._print(f'Removing invalid csv log {csv_logger_file}')
            csv_logger.csv_file.close()
            try:
                os.remove(csv_logger_file)
            except PermissionError:
                self._print('File could not be removed')
            except FileNotFoundError:
                self._print('File was not created')

        # Reduce learning rate
        if self._reduce_lr_on_plateau['enabled']:
            _callbacks.append(ReduceLROnPlateau(
                monitor=self._reduce_lr_on_plateau['monitor'],
                factor=self._reduce_lr_on_plateau['factor'],
                patience=self._reduce_lr_on_plateau['patience'],
                verbose=self._print_enabled,
                mode=self._reduce_lr_on_plateau['mode'],
                min_delta=self._reduce_lr_on_plateau['min_delta'],
                cooldown=self._reduce_lr_on_plateau['cooldown'],
                min_lr=self._reduce_lr_on_plateau['min_lr']
            ))
            self._print('[Callback] Using reduce learning rate on plateau, '
                        'patience {0} mode {1}, factor {2}'.format(self._reduce_lr_on_plateau['patience'],
                                                                   self._reduce_lr_on_plateau['mode'],
                                                                   self._reduce_lr_on_plateau['factor']))

        if len(_callbacks) > 0:
            self._print('')

        # Time callback
        time_history_callback = TimeHistory()
        _callbacks.append(time_history_callback)

        def exception_callbacks() -> None:
            """
            Function executed if an exception happens.
            """
            if not continue_train:
                self.reset_weights()
            remove_checkpoint_empty()
            remove_csv_log()
            gc.collect()

        # Assert stateful metrics
        mlabels = self.get_metric_names()
        for i in self.get_stateful_metrics_names():
            assert i in mlabels, f'Stateful metric <{i}> does not exists in current metrics'

        # Assemble train vectors
        train_x = self._format_tuple(xtrain)
        train_y = self._format_tuple(ytrain)

        # Compute shape
        xtrain_shape = self._make_shape(xtrain)
        ytrain_shape = self._make_shape(ytrain)

        self._print(f'Train initialized at: {date_train}')
        self._print(f'Train shape: {xtrain_shape} -> {ytrain_shape}')
        self._print(f'Train shuffle: {shuffle}')
        self._print(f'Train epochs: {epochs} with batch size {batch_size}')
        self._print(f'Train validation partition: {val_split * 100:.1f}%')
        self._print('')

        self._print(f'Metrics: {self.get_metric_names()}')
        self._print(f'Stateful metrics: {self.get_stateful_metrics_names()}')

        initial_epoch = 0
        if continue_train:
            initial_epoch = self._train_metadata['epochs']
            self._print(f'Starting from epoch: {initial_epoch}')

        # Fit model
        tini_fit = time.time()
        try:
            if not use_custom_fit:
                self._print('Using: Standard fit by tensor')
                self._history = self._model.fit(
                    x=train_x,
                    y=train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=not self._use_tqdm_notebook and self._use_fit_logger and self._print_enabled,
                    callbacks=_callbacks,
                    validation_split=val_split,
                    shuffle=shuffle  # Boolean (whether to shuffle the training data before each epoch)
                ).history
            else:
                self._print('Using: Custom fit by tensor')
                self._history = self._custom_fit_tensor(
                    x=train_x,
                    y=train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=not self._use_tqdm_notebook and self._use_fit_logger and self._print_enabled,
                    callbacks=_callbacks,
                    validation_split=val_split,
                    shuffle=shuffle  # Boolean (whether to shuffle the training data before each epoch)
                ).history
        except (KeyboardInterrupt, KeyError) as e:  # Keyboard for user, Key for early
            # traceback.print_exc()
            self._print(f'Exception: {e}')
            self._print('Train ended by user')

            def _reset() -> None:
                """
                Reset and return.
                """
                if continue_train:
                    return
                self.reset_weights()
                self._is_trained = False
                gc.collect()

            if not self._use_csv_logger and not continue_train:
                return _reset()

            # Load history from logger
            try:
                self._is_trained = True
                self._load_history_from_csv(csv_logger_file)
                self._print(f"Saved {self._train_metadata['epochs']} epochs, setting model as trained")
            except AssertionError:
                self._train_metadata['epochs'] = initial_epoch

            remove_checkpoint_empty()
            if self._train_metadata['epochs'] == initial_epoch and not continue_train:
                remove_csv_log()
                _reset()
        except tf.errors.ResourceExhaustedError:
            # traceback.print_exc()
            self._print('Out of memory, try reducing the batch size or resizing the database')
            return exception_callbacks()
        except Exception as e:
            traceback.print_exc()
            self._print(f'Uncaught Exception: {e}')
            return exception_callbacks()

        # Compute train fit time and epoch avg
        train_fit_time = time.time() - tini_fit
        if len(time_history_callback.times) > 1:
            time_history_callback.times.pop(0)  # Ignore first
        train_fit_avg_epoch = sum(time_history_callback.times) / max(1, len(time_history_callback.times))

        # Convert history to floats
        for k in self._history.keys():
            hv: List[float] = []
            for i in range(len(self._history[k])):
                hv.append(float(self._history[k][i]))
            self._history[k] = hv

        if len(self._history.keys()) > 0:
            # Extend history
            if continue_train:
                assert prev_history.keys() == self._history.keys()
                for k in prev_history.keys():
                    for j in self._history[k]:
                        prev_history[k].append(j)
                del self._history
                self._history = prev_history

            self._train_metadata['epochs'] = len(self._history[list(self._history.keys())[0]])
            self._is_trained = True
            if compute_metrics:
                self._compute_metrics(xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest)

        else:
            self._train_metadata['epochs'] = initial_epoch
            self._print('Could not compute metrics as epoch are null')
            remove_csv_log()

        # If empty, remove checkpoints
        remove_checkpoint_empty()

        train_data = {}
        train_time = time.time() - tini
        if continue_train:
            train_time += self._train_metadata['train_time']
            train_fit_time += self._train_metadata['train_fit_time']
            train_data = self._train_metadata[_RUNTIME_METADATA_KEY]
        self._print(f'Process finished in: {datetime.timedelta(seconds=train_time)}')

        # Save train config data
        self._train_metadata = {
            'batch_size': batch_size,
            'compute_metrics': compute_metrics,
            'continue_train': continue_train,
            'continue_train_count': continue_train_count,
            'csv_logger_file': csv_logger_file,
            'custom_train_fit': use_custom_fit,
            'epochs': self._train_metadata['epochs'],  # As dict is rewritten
            'max_epochs': epochs,
            'model_checkpoint_path': model_checkpoint_path,
            'shuffle': shuffle,
            'tensorboard_log': tensorboard_log,
            'train_date': date_train,
            'train_shape_x': xtrain_shape,
            'train_shape_y': ytrain_shape,
            'train_time': train_time,
            'train_fit_time': train_fit_time,
            'train_fit_epoch_avg_time': train_fit_avg_epoch,
            'validation_split': val_split,
            _RUNTIME_METADATA_KEY: train_data
        }
        self._extend_train_metadata()

        # Collect memory
        time.sleep(5)
        gc.collect()

    def _extend_train_metadata(self) -> None:
        """
        Extend train metadata values.
        """
        metak = {
            'max_epochs': self._train_metadata['epochs'],
            'train_date': datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S'),
            'train_time': 0,  # Model was not trained (loaded by file)
            'train_fit_time': 0,
            'train_fit_epoch_avg_time': 0
        }
        for k in metak.keys():
            if k not in self._train_metadata.keys():
                self._train_metadata[k] = metak[k]
        if _RUNTIME_METADATA_KEY not in self._train_metadata.keys():
            self._train_metadata[_RUNTIME_METADATA_KEY] = {}

    def _custom_train_function(self, inputs) -> List[float]:
        """
        Custom train function.
        inputs: x + y + sample_weights (1 list for each output).

        :param inputs: Train input
        :return: Train metrics
        """
        raise RuntimeError('Model does not support custom train function')

    def _custom_val_function(self, inputs) -> List[float]:
        """
        Custom validation function.
        inputs: x + y + sample_weights (1 list for each output).

        :param inputs: Train input
        :return: Validation metrics
        """
        raise RuntimeError('Model does not support custom validation function')

    def _custom_epoch_finish_function(self, num_epoch: int) -> None:
        """
        Function triggered once each epoch finished.

        :param num_epoch: Number of the epoch
        """
        return

    def _add_custom_metric(self, metric: str) -> None:
        """
        Add custom metrics.

        :param metric: Metric name, used by custom functions
        """
        if metric in self._custom_metrics:
            raise ValueError(f'Custom metric <{metric}> already in model metrics')
        self._custom_metrics.append(metric)

    # noinspection PyProtectedMember
    def _custom_fit_tensor(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs
    ) -> 'History':
        """
        Custom model fit on batches.

        :param x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
                - A generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample weights)`.
                - None (default) if feeding from framework-native
                  tensors (e.g. TensorFlow data tensors).
        :param y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                If `x` is a generator, or `keras.utils.Sequence` instance,
                `y` should not be specified (since targets will be obtained
                from `x`).
        :param batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, generators, or `Sequence` instances
                (since they generate batches).
        :param epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training and validation
                (if ).
                See [callbacks](/callbacks).
        :param validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
                This argument is not supported when `x` is a generator or
                `Sequence` instance.
        :param validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                    - tuple `(x_val, y_val)` of Numpy arrays or tensors
                    - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                    - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
        :param shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
        :param class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
        :param sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`. This argument
                is not supported when `x` generator, or `Sequence` instance,
                instead provide the sample_weights as the third element of `x`.
        :param initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
        :param steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
        :param validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.
        :param validation_steps: Only relevant if `validation_data` is provided
                and is a generator. Total number of steps (batches of samples)
                to draw before stopping when performing validation at the end
                of every epoch.
        :param validation_freq: Only relevant if validation data is provided. Integer
                or list/tuple/set. If an integer, specifies how many training
                epochs to run before a new validation run is performed, e.g.
                `validation_freq=2` runs validation every 2 epochs. If a list,
                tuple, or set, specifies the epochs on which to run validation,
                e.g. `validation_freq=[1, 2, 10]` runs validation at the end
                of the 1st, 2nd, and 10th epochs.
        :param max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
        :param workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
        :param use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        :param kwargs: Optional keyword arguments

        :return:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        :raises:
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)

        # Legacy support
        if 'nb_epoch' in kwargs:
            e = 'The `nb_epoch` argument in `fit` has been renamed `epochs`.'
            warnings.warn(e, stacklevel=2)
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        if x is None and y is None and steps_per_epoch is None:
            raise ValueError('If fitting from data tensors, '
                             'you should specify the `steps_per_epoch` '
                             'argument.')

        batch_size = self._model._validate_or_infer_batch_size(
            batch_size, steps_per_epoch, x)
        val_inputs = None

        # Case 1: generator-like. Input is Python generator,
        # or Sequence object, or iterator
        if training_utils.is_generator_or_sequence(x):
            training_utils.check_generator_arguments(
                y, sample_weight, validation_split=validation_split)
            self._print('Using: Custom fit by generator')
            return self._custom_fit_generator(
                x,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
                validation_freq=validation_freq,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                shuffle=shuffle,
                initial_epoch=initial_epoch
            )

        # Case 2: Symbolic tensors or Numpy array-like
        x, y, sample_weights = self._model._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            batch_size=batch_size
        )

        # Prepare validation data
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._model._standardize_user_data(
                val_x, val_y,
                sample_weight=val_sample_weight,
                batch_size=batch_size)
            if self._model._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            if any(is_tensor(t) for t in x):
                raise ValueError(
                    'If your data is in the form of symbolic tensors, '
                    'you cannot use `validation_split`.')
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(int(x[0].shape[0]) * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                slice_arrays(sample_weights, 0, split_at),
                slice_arrays(sample_weights, split_at))
            if self._model._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_steps:
            do_validation = True
            if self._model._uses_dynamic_learning_phase():
                val_inputs = [0]

        # Prepare input arrays and training function
        if self._model._uses_dynamic_learning_phase():
            fit_inputs = x + y + sample_weights + [1]
        else:
            fit_inputs = x + y + sample_weights

        # Prepare display labels
        out_labels = self.get_metric_names()

        if do_validation:
            val_function = self._custom_val_function
        else:
            val_function = None
            val_inputs = []

        # Delegate logic to fit_loop
        # self._print('Out labels: {0}'.format(', '.join(out_labels)))
        return fit_loop(
            model=self._model,
            model_metrics_names=self.get_metric_names(),
            model_stateful_metrics_names=self.get_stateful_metrics_names(),
            fit_function=self._custom_train_function,
            fit_inputs=fit_inputs,
            out_labels=out_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            val_function=val_function,
            val_inputs=val_inputs,
            shuffle=shuffle,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            epoch_finish_function=self._custom_epoch_finish_function
        )

    def get_metric_names(self) -> List[str]:
        """
        :return: Returns model metrics names
        """
        if not self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_NOT_COMPILED)
        out_labels: List[str] = list(self._model.metrics_names)
        for m in self._custom_metrics:
            out_labels.append(m)
        return out_labels

    def get_total_metrics(self) -> int:
        """
        :return: Returns the total metrics
        """
        return len(self._model.metrics_names)

    def get_stateful_metrics_names(self) -> List[str]:
        """
        :return: Returns the stateful metric names
        """
        if not self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_NOT_COMPILED)
        if isinstance(self._custom_stateful_metrics, list):
            return self._custom_stateful_metrics
        return self._model.metrics_names[1:]

    def _custom_fit_generator(self, *args, **kwargs) -> 'History':
        """
        Custom fit generator.

        :param args: Optional non-keyword arguments
        :param kwargs: Optional keyword arguments
        :return: History
        """
        raise RuntimeError('Model does not support fit on generator')

    def _compute_metrics(self, xtrain, xtest, ytrain, ytest) -> None:
        """
        Compute model metrics.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self._print('Computing training/testing metrics:')
        o_verbose = self._verbose
        self._verbose = True
        try:
            self._metric_train = self.evaluate(x=xtrain, y=ytrain)
        except KeyboardInterrupt:
            self._print('\nTrain evaluation process interrupted by user')
            self._metric_train = []
        try:
            self._metric_test = self.evaluate(x=xtest, y=ytest)
        except KeyboardInterrupt:
            self._print('\nTest evaluation process interrupted by user')
            self._metric_test = []
        if not isinstance(self._metric_test, list):
            # noinspection PyTypeChecker
            self._metric_test = [self._metric_test]
        if not isinstance(self._metric_train, list):
            # noinspection PyTypeChecker
            self._metric_train = [self._metric_train]
        self._verbose = o_verbose
        self.print_metrics()

    def _register_train_data(self, key: str, value: Union[int, float, str, List]):
        """
        Add data to train metadata.
        This data will be wiped after training.

        :param key: Data key
        :param value: Data value
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if isinstance(value, list):
            for v in value:
                assert isinstance(v, (int, float, str)), f'Invalid data type {v} from value list'
        if _RUNTIME_METADATA_KEY not in self._train_metadata:
            self._train_metadata[_RUNTIME_METADATA_KEY] = {}
        self._train_metadata[_RUNTIME_METADATA_KEY][key] = value

    def _exists_in_train_data(self, key: str) -> bool:
        """
        Check if key exists in train data.

        :param key: Key
        :return: True if exists
        """
        if _RUNTIME_METADATA_KEY not in self._train_metadata:
            self._train_metadata[_RUNTIME_METADATA_KEY] = {}
        return key in self._train_metadata[_RUNTIME_METADATA_KEY].keys()

    def _get_train_data(self, key: str) -> Union[int, float, str, List]:
        """
        Get train data.

        :param key: Key
        :return: Data value or KeyError
        """
        if not self._exists_in_train_data(key):
            raise KeyError(f'Key <{key}> does not exists on train data')
        return self._train_metadata[_RUNTIME_METADATA_KEY][key]

    def reset_weights(self, print_status: bool = True) -> None:
        """
        Reset model weights.

        :param print_status: If true, print resetting status
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if not self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_NOT_COMPILED)
        self._is_trained = False
        if print_status:
            self._print('Resetting weights')
        self._model.reset_states()
        _reset_weight(self._model)

    def get_name(self, formatted: bool = False) -> str:
        """
        Returns the model name.

        :param formatted: Returns formatted name
        :return: Name
        """
        if formatted:
            return self._name_formatted
        return self._name

    def export_history_to_csv(self, file_csv: str = '') -> None:
        """
        Export history to file.

        :param file_csv: Output csv file
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        if file_csv == '':
            train_date: str = self._train_metadata['train_date']
            train_date = train_date.replace(' ', '_').replace(':', '-').replace('/', '-')
            file_csv = '{0}{3}{1}_{2}.csv'.format(os.path.join(self._path, _PATH_LOGS), self._name_formatted,
                                                  train_date,
                                                  os.path.sep)
            self._print(f'Exporting to file: {file_csv}')
        fcsv = open(file_csv, 'w')
        firstl: List[str] = ['epoch']
        for k in self._history.keys():
            firstl.append(k)
        fcsv.write(','.join(firstl) + '\n')  # Header
        for i in range(self.get_total_epochs()):
            kline: List[str] = [str(i)]
            for k in self._history.keys():
                kline.append(str(self._history[k][i]))
            fcsv.write(','.join(kline) + '\n')
        fcsv.close()

    def print_metrics(self) -> None:
        """
        Print metrics.
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)

        keys: List[str] = []
        if self._production:
            for k in self._history.keys():
                keys.append(k.replace('val_', ''))
        else:
            for k in self.get_metric_names():
                keys.append(k.replace('val_', ''))

        def print_results_metrics(m: List[float]):
            """
            Print metrics.
            """
            _max_ln: int = 0
            for i in range(len(m)):
                _max_ln = max(_max_ln, len(keys[i]))

            for i in range(len(m)):
                self._print(f'\t{keys[i].ljust(_max_ln)}: {m[i]}')

        if len(self._metric_train) > 0:
            self._print('Train metrics:')
            print_results_metrics(self._metric_train)
        if len(self._metric_test) > 0:
            self._print('Test metrics:')
            print_results_metrics(self._metric_test)

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """
        Public predict model at matrix/vector x.

        :param x: X to eval
        :return: y=Ƒ(x) where Ƒ is the model
        """
        pass

    @abstractmethod
    def evaluate(self, x: Any, y: Any) -> Union[List[float], float]:
        """
        Public evaluate model for inputs and outputs.

        :param x: x to evaluate
        :param y: y to evaluate
        :return: Model evaluation
        """
        pass

    def _model_predict(self, x: Union['np.ndarray', List['np.ndarray'], Tuple['np.ndarray']]) -> Any:
        """
        Predict model at matrix/vector x.

        :param x: X to eval
        :return: y=Ƒ(x) where F is the model
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        return self._model.predict(x=x)

    def _model_evaluate(
            self,
            x: Union['np.ndarray', List['np.ndarray'], Tuple['np.ndarray']],
            y: Union['np.ndarray', List['np.ndarray'], Tuple['np.ndarray']]
    ) -> List[float]:
        """
        Evaluate model at matrix/vector x.

        :param x: x to evaluate
        :param y: y to evaluate
        :return: Model evaluation
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        return self._model.evaluate(x=x, y=y, verbose=self._verbose)

    def info(self) -> None:
        """
        Display model info.
        """
        self._model.summary()

    @staticmethod
    def clear_session() -> None:
        """
        Call keras clear session.
        """
        # print('Clearing model session')
        clear_session()
        gc.collect()

    def get_train_history(self) -> Dict[str, List[float]]:
        """
        :return: History of training
        """
        if len(self._history) == 0:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        return self._history

    def get_total_epochs(self) -> int:
        """
        :return: Get train total epochs
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        return self._train_metadata['epochs']

    def compile(
            self,
            optimizer: Union[str, 'Optimizer', 'OptimizerV2'],
            loss: Union[Union[str, Callable], Dict[str, Union[str, Callable]], List[Union[str, Callable]]],
            metrics: Optional[Union[Union[Optional[Union[str, Callable]], List[Optional[Union[str, Callable]]]], Dict[
                str, Union[Optional[Union[str, Callable]], List[Optional[Union[str, Callable]]]]]]] = None,
            loss_weights: Optional[Union[Dict[str, Union[int, float]], List[Union[int, float]]]] = None,
            as_list: bool = False
    ) -> None:
        """
        Compile model.

        :param optimizer: Optimizer
        :param loss: Loss
        :param metrics: Metrics
        :param loss_weights: Weights for each loss
        :param as_list: Use lists instead of dicts (used for concatenated models)
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_COMPILED)
        if self._is_trained:
            raise RuntimeError(_ERROR_MODEL_TRAINED)

        # Use dict
        if not as_list:
            reqk: str = ', '.join(self._output_layers)
            total_out: int = len(self._output_layers)

            # Transform to dict in case of total_out>1
            if total_out > 1:
                if isinstance(loss, (str, Callable)):
                    self._print(f'Compile: Setting the same loss for each output ({total_out})')
                    vloss = loss
                    loss = {}
                    for k in self._output_layers:
                        loss[k] = vloss
                if isinstance(metrics, (str, Callable, list, type(None))):
                    if metrics is not None:
                        self._print(f'Compile: Setting the same metric for each output ({total_out})')
                    vmetrics = metrics
                    metrics = {}
                    for k in self._output_layers:
                        metrics[k] = vmetrics

            # Check loss
            if isinstance(loss, dict):
                assert len(loss.keys()) == total_out, \
                    'Invalid number of output loss losses, correct: {0}, ' \
                    'required keys <{1}>'.format(total_out, reqk)
                for k in loss.keys():
                    assert k in self._output_layers, \
                        f'Output layer <{k}> at loss does not exist, current <{reqk}>'
                    assert isinstance(loss[k], (str, Callable)), \
                        f'Value at loss key <{k}> must be a string or a function'
            else:
                if total_out > 1:
                    _msg = 'As model has more than one output loss ' \
                           'must be defined as a dict with keys: <{0}>'.format(', '.join(self._output_layers))
                    raise ValueError(_msg)
                assert isinstance(loss, (str, Callable)), \
                    'Loss must be a string or a function'

            # Loss weights
            if loss_weights is None:
                if total_out > 1:
                    self._print('Compile: Loss weights should be defined as number of outputs is greater than one, '
                                'using default loss_weights 1.0 for each output')
                    loss_weights = {}
                    for k in self._output_layers:
                        loss_weights[k] = 1.0
            else:
                assert total_out > 1, \
                    'Loss weights cannot be defined if the model only has one output'
                assert len(loss_weights.keys()) == len(self._output_layers), \
                    f'Number of loss weight keys must be the same as the number of model outputs ({total_out})'
                for k in loss_weights.keys():
                    assert isinstance(loss_weights[k], (int, float)), \
                        f'Loss weight at key <{k}> must be a number'
                    loss_weights[k] = float(loss_weights[k])

            # Check metrics
            if isinstance(metrics, str):
                metrics = [metrics]
            init_metrics: dict
            if metrics is not None:
                init_metrics = metrics.copy()
            else:
                init_metrics = {}
            if isinstance(metrics, dict):
                if len(metrics.keys()) < total_out:
                    for k in self._output_layers:
                        if k not in metrics.keys():
                            metrics[k] = None
                assert len(metrics.keys()) == total_out, \
                    'Invalid number of output metrics, correct: {0}, ' \
                    'required keys: <{1}>'.format(total_out, reqk)
                for k in metrics.keys():
                    assert k in self._output_layers, \
                        f'Output layer <{k}> at metric does not exist, model defined: <{reqk}>'
                for k in self._output_layers:
                    if metrics[k] is None:
                        del metrics[k]
                        continue
                    assert isinstance(metrics[k], (str, Callable, list)), \
                        f'Value at metric key <{k}> must be a string, None, a function or a list of metrics'
                    if isinstance(metrics[k], list):
                        assert len(metrics[k]) > 0, f'Metrics at key list <{k}> cannot be empty'
                        unique = []
                        for i in range(len(metrics[k])):
                            assert isinstance(metrics[k][i], (str, Callable)), \
                                f'Value at metric key <{k}> pos <{i}> must be a string or a function'
                            if metrics[k][i] == loss[k]:
                                self._print('Compile: Loss <{0}> should not be used as a metric for output <{1}>, '
                                            'removing from the list'.format(metrics[k][i], k))
                            else:
                                unique.append(metrics[k][i])
                        metrics[k] = unique
                    else:
                        if metrics[k] == loss[k]:
                            self._print('Compile: Loss <{0}> should not be used as a metric for output <{1}>, '
                                        'removing from the list'.format(metrics[k], k))
                            del metrics[k]
            elif metrics is None:
                if len(self._output_layers) != 1:
                    metrics = {}
                    for k in self._output_layers:
                        metrics[k] = None
            else:
                if total_out > 1:
                    raise ValueError('As model has more than one output metrics '
                                     'must be defined as a dict with keys: <{0}>'.format(reqk))
                assert isinstance(metrics, (str, Callable, list, type(None))), \
                    'Metric must be a string, a function or a List of metrics, or None'
                if isinstance(metrics, list):
                    assert len(metrics) > 0, \
                        'Metrics list cannot be empty. Use None instead'
                    for i in range(len(metrics)):
                        assert isinstance(metrics[i], (str, Callable, type(None))), \
                            f'Metric list pos <{i}> must be a string or a function'
                    # Check loss is being used as a metric too
                    lossm = loss
                    if isinstance(loss, dict):
                        lossm = loss[list(loss.keys())[0]]
                    if lossm in metrics:
                        self._print('Compile: Loss <{0}> should not be used as a '
                                    'metric, removing from the list'.format(lossm))
                        for k in range(len(metrics)):
                            if metrics[k] == lossm:
                                metrics.pop(k)
                                break
                        if len(metrics) == 0:
                            metrics = None

        # Use lists
        else:
            if isinstance(loss, str):
                loss = [loss]
            assert isinstance(loss, list), 'Lost must be a list or string'
            assert isinstance(loss_weights, (type(None), list))
            assert len(loss) == len(self._output_layers), \
                'Loss length must be the same as the number of outputs'
            if loss_weights is not None:
                assert len(loss_weights) == len(loss), \
                    'Loss weights length must be the same as the number of losses'
                for i in range(len(loss_weights)):
                    assert isinstance(loss_weights[i], (int, float)), \
                        f'Loss weight at position <{i}> must be a number'
            assert isinstance(metrics, (type(None), str, Callable, list))

            init_metrics = metrics
            if isinstance(metrics, list):
                for i in range(len(metrics)):
                    assert isinstance(metrics[i], (str, Callable, type(None))), \
                        f'Metric list pos <{i}> must be a string or a function'
                # Check loss is being used as a metric too
                for ls in loss:
                    if ls in metrics:
                        self._print(f'Loss <{ls}> should not be used as a metric, removing from the list')
                        for k in range(len(metrics)):
                            if metrics[k] == ls:
                                metrics.pop(k)
                                break
                        if len(metrics) == 0:
                            metrics = None

        # Compile the model
        self._model.compile(
            optimizer=optimizer,  # https://keras.io/optimizers/
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics
        )
        self._is_compiled = True
        self._compile_config = {
            'optimizer': optimizer,
            'loss': loss,
            'loss_weights': loss_weights,
            'metrics': init_metrics
        }

    def remove_from_history(self, key: str) -> None:
        """
        Remove metric from history.

        :param key: History
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if key in self._history.keys():
            del self._history[key]
        else:
            raise KeyError(f'History key <{key}> does not exists')
        if 'val_' + key in self._history.keys():  # Validation
            del self._history['val_' + key]

    def get_train_time(self) -> float:
        """
        :return: Returns total training time of the model
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        return self._train_metadata['train_time']

    def get_train_fit_time(self) -> Tuple[float, float]:
        """
        :return: Returns total training fit time of the model, and the average time per epoch
        """
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        return self._train_metadata['train_fit_time'], self._train_metadata['train_fit_epoch_avg_time']

    def _get_compile_config(self) -> Dict[str, Any]:
        """
        :return: Return compile config list as a sum of strings or numbers
        """

        def _parse_object(obj: Any) -> Union[str, float, int, Dict[str, Any]]:
            """
            Convert object.

            :param obj: Object
            :return: String from object
            """
            if ' at ' in str(obj):  # If class or function
                o = str(obj)
                o = o.split(' at ')[0] + '>'
                # o = o.replace(' object', '')
                if isinstance(obj, Optimizer) and False:  # Test
                    return {
                        'object': o,
                        'config': obj.get_config()
                    }
                else:
                    return o
            elif isinstance(obj, dict):
                return _parse_dict(obj)
            else:
                return obj

        def _parse_dict(d: Dict[str, Any]) -> Dict:
            """
            Create a string dict from source.

            :param d: Input dict
            :return: Output string dict
            """
            newd: Dict[str, Any] = {}
            for k in d.keys():
                if isinstance(d[k], dict):
                    newd[k] = _parse_dict(d[k])
                elif isinstance(d[k], list):
                    newl: List[Any] = []
                    for j in d[k]:
                        newl.append(_parse_object(j))
                    newd[k] = newl
                else:
                    newd[k] = _parse_object(d[k])
            return newd

        return _parse_dict(self._compile_config)

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        Custom session save data. Used by extended methods.

        :param filename: Filename of the session
        :param data: Data to be saved as a dict
        """
        return

    def save_session(self, filename: str = '', description: str = '', save_weights: bool = True) -> None:
        """
        Save model session.

        :param filename: Filename of the session
        :param description: Session file description
        :param save_weights: Save model weights
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if not self._is_trained:
            raise RuntimeError(_ERROR_MODEL_NOT_TRAINED)
        if filename == '':
            filename = self._name_formatted
        filename = os.path.splitext(filename)[0]
        self._print(f'Saving session to: {filename}')

        # Files
        file_weights: str = filename + '_weights.h5'
        file_model_arch: str = filename + '_architecture.dat'
        file_session: str = filename + '.json'

        # Save model weights
        if save_weights:
            self._model.save_weights(file_weights)

        # Save model architectures
        model_json = self._model.to_json(indent=2)
        with open(file_model_arch, 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)

        def _sorted_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """
            :return: Returns sorted session dict data
            """
            sork = list(d.keys())
            sork.sort()
            w = {}
            for k in sork:
                w[k] = d[k]
            return w

        # Generate json model
        with open(file_session, 'w', encoding='utf-8') as fp:
            data = {

                # File
                'version': _SESSION_EXPORT_VERSION,
                'save_date': datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'),

                # Session
                'description': description,
                'session_data': _sorted_dict(self._session_data),

                # Model basics
                'class': self.__class__.__name__,
                'class_version': self._version,
                'compile_config': self._get_compile_config(),
                'custom_metrics': self._custom_metrics,
                'custom_stateful_metrics': self._custom_stateful_metrics,
                'name': self._name,
                'output_layers': self._output_layers,
                'test_split': round(self._test_split, 4),

                # Model num. of weights
                'non_trainable_weights': count_params(self._model.non_trainable_weights),
                'trainable_weights': count_params(self._model.trainable_weights),

                # Model hashes
                'hash_model': self._get_model_hash(),
                'hash_session_arch': file_md5(file_model_arch),
                'hash_session_weights': file_md5(file_weights),

                # Callbacks data
                'early_stopping': _sorted_dict(self._early_stopping),
                'model_checkpoint': _sorted_dict(self._model_checkpoint),
                'reduce_lr_on_plateau': _sorted_dict(self._reduce_lr_on_plateau),
                'use_csv_logger': self._use_csv_logger,
                'use_tensorboard': self._use_tensorboard,
                'use_tqdm_notebook': self._use_tqdm_notebook,
                'verbose': self._verbose,

                # Train
                'history_keys': list(self._history.keys()),
                'metric_test': self._metric_test,
                'metric_train': self._metric_train,
                'train_metadata': _sorted_dict(self._train_metadata)

            }

            # Use custom save data
            self._custom_save_session(filename=filename, data=data)

            # Save history, last as this is a large data
            data['history'] = self._history

            json.dump(data, fp, indent=2)
            self._loaded_session = {
                'description': description,
                'file': filename,
                'last': 'save',
                'train_date': self._train_metadata['train_date']
            }

    def _get_model_hash(self) -> str:
        """
        :return: Returns the current model hash
        """
        if not self._is_compiled:
            raise RuntimeError(_ERROR_MODEL_NOT_COMPILED)
        trainable_count = str(count_params(self._model.trainable_weights))
        non_trainable_count = str(count_params(self._model.non_trainable_weights))
        h = hashlib.md5()
        h.update(trainable_count.encode())
        h.update(non_trainable_count.encode())
        h.update(_normalize_arch_json(self._model.to_json(indent=2)).encode())
        return h.hexdigest()

    def _custom_load_session(
            self,
            filename: str,
            asserts: bool,
            data: Dict[str, Any],
            check_hash: bool
    ) -> None:
        """
        Custom load session.

        :param filename: Filename of the session
        :param asserts: If true, session is in assert mode
        :param data: Data from session
        :param check_hash: Checks file hash
        """
        return

    def load_session(
            self,
            filename: str,
            override_model: bool = False,
            override_callbacks: bool = False,
            check_hash: bool = True
    ) -> 'GenericModel':
        """
        Load model from session.

        :param filename: Filename of the session
        :param override_model: Reload model from file
        :param override_callbacks: Override model callbacks
        :param check_hash: Checks file hash
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        t0 = time.time()  # Init time

        if filename == '':
            filename = self._name_formatted

        filename = os.path.splitext(filename)[0]
        if '.json' in filename:
            filename = filename.replace('.json', '')

        # Check file exists
        file_arch = filename + '_architecture.dat'
        file_session = filename + '.json'
        file_weights = filename + '_weights.h5'

        assert os.path.isfile(file_session), f'Session file <{file_session}> does not exist'
        assert os.path.isfile(file_weights), f'Session weight file <{file_weights}> does not exist'
        assert os.path.isfile(file_arch), f'Session model architectures file <{file_arch}> does not exist'

        data: 'Dict'
        with open(file_session, 'r') as fp:
            data = json.load(fp)

        # Check export version
        _ver = 'Outdated session export version, needed {0}, ' \
               'current {1}'.format(_SESSION_EXPORT_VERSION, data['version'])
        assert data['version'] == _SESSION_EXPORT_VERSION, _ver

        # Assert object class
        _class = 'Session model class <{0}> is different from current ' \
                 'model class <{1}>'.format(data['class'], self.__class__.__name__)
        assert data['class'] == self.__class__.__name__, _class

        # Assert class version
        _classv = 'Session class version <{0}> is different from current ' \
                  'model class version <{1}>'.format(data['class_version'], self._version)
        assert data['class_version'] == self._version, _classv

        # Check model architecture
        model_equal: bool = data['hash_model'] == self._get_model_hash()
        if not override_model and check_hash:
            if not model_equal:
                _err = 'Model has changed, printing difference between architectures'

                # Get current architectures
                current_arch = self._model.to_json(indent=2)

                # Calculate the difference between architectures files
                arch_file = open(file_arch, 'r')
                modelj: str = ''
                for i in arch_file:
                    modelj += i
                arch_diff: Iterator[str] = difflib.unified_diff(
                    _normalize_arch_json(modelj).split('\n'),
                    _normalize_arch_json(current_arch).split('\n')
                )

                # Store diff
                arch_diffl: List[str] = []
                for a in arch_diff:
                    arch_diffl.append(a)

                if len(arch_diffl) > 0:
                    # Console out
                    self._print(_err)
                    self._print('Architecture diff:')
                    for line in arch_diffl:
                        self._print('\t' + line)
                    assert model_equal, 'Model hash changed'
                else:
                    _err = 'Hash model changed but the architectures are the same, session ' \
                           'update needed with .session_update()'
                    # warnings.warn(_err)
                    if self._check_compilation:  # This is a difference mostly on compile
                        self._print(_err)

        # Model hashes
        if check_hash:
            assert data['hash_session_weights'] == file_md5(file_weights), 'File session weights hash changed'
            assert data['hash_session_arch'] == file_md5(file_arch), 'File session model architecture hash changed'
            # assert 0 < data['test_split'] < 1, 'Invalid data test split value'

        # Assert training epoch
        assert data['train_metadata']['epochs'] <= data['train_metadata']['max_epochs'], 'Invalid epochs'

        # Check output layer names are the same
        outlay: List[str] = data['output_layers']
        assert len(outlay) == len(
            self._output_layers), 'Different number of outputs {0}⟶{1} not supported'.format(
            self._output_layers, outlay)
        for k in range(len(outlay)):
            assert outlay[k] == self._output_layers[k], \
                'Expected key <{0}> at position <{1}> of output ' \
                'layer list, current <{2}>'.format(outlay[k], k, self._output_layers[k])

        # Check custom metrics are the same
        assert len(data['custom_metrics']) == len(self._custom_metrics), \
            'Custom metrics length changed'
        for k in range(len(data['custom_metrics'])):
            assert data['custom_metrics'][k] == self._custom_metrics[k], \
                'Expected <{0}> at position <{1}> of ' \
                'custom metric list, given <{2}>'.format(self._custom_metrics[k], k, data['custom_metrics'][k])

        # Check custom stateful metrics are the same
        if self._custom_stateful_metrics is None:
            assert data['custom_stateful_metrics'] is None, 'Custom stateful metrics must be None'
        else:
            for k in range(len(data['custom_stateful_metrics'])):
                assert data['custom_stateful_metrics'][k] == self._custom_stateful_metrics[k], \
                    'Expected <{0}> at position <{1}> of ' \
                    'custom stateful metric list, given <{2}>'.format(
                        self._custom_stateful_metrics[k], k, data['custom_stateful_metrics'][k])

        # Custom asserts
        self._custom_load_session(filename=filename, asserts=True, data=data, check_hash=check_hash)

        # If compile config is different print a message
        current_compile_config: Dict[str, Any] = self._get_compile_config()
        if current_compile_config != data['compile_config'] and len(current_compile_config.keys()) > 0 \
                and self._check_compilation:
            _compile_msg = 'Compile configuration from session is different from the current model'
            # warnings.warn(_compile_msg)
            self._print(_compile_msg + ':')
            for k in current_compile_config.keys():
                if str(current_compile_config[k]) != str(data['compile_config'][k]):
                    self._print(f'\tDifferences in <{k}>:')
                    self._print(f'\t\tCurrent: {current_compile_config[k]}')
                    self._print(f"\t\tSession: {data['compile_config'][k]}")

        # Reload model if the model is not equal
        if override_model:
            if not model_equal:
                e = f'Overriding model from {file_arch} model architectures file'
                warnings.warn(e)
                self._print('This model must be compiled again, then reload the session')
                with open(file_arch, 'r') as json_file:
                    architecture = json.load(json_file)
                self._model = model_from_json(json.dumps(architecture))
                self._is_compiled = False
                return self
            else:
                self._print('The model architectures has not been overwritten as the hash is the same')

        # self._print('Loaded session:\t{0}'.format(filename))
        self._print(f"Model name:\t{data['name']}")
        if len(data['description']) > 0:
            self._print(f"Description:\t{data['description']}")
        self._print(f"Save date:\t{data['save_date']}")
        # self._print('')
        if data['train_metadata']['train_time'] == 0:
            self._print('Train:\t\tLoaded from file')
        else:
            self._print(f"Train time:\t{datetime.timedelta(seconds=data['train_metadata']['train_time'])}")
        if 'train_fit_time' not in data['train_metadata'].keys():
            data['train_metadata']['train_fit_time'] = 0
            data['train_metadata']['train_fit_epoch_avg_time'] = 0
        self._print(f"Train date:\t{data['train_metadata']['train_date']}")
        self._print(f"Train epochs:\t{data['train_metadata']['epochs']}")

        # Load weights
        self._load_weights_from_file(file_weights)

        # Save others
        self.set_name(data['name'])
        self._history = data['history']
        self._is_trained = True
        self._metric_test = data['metric_test']
        self._metric_train = data['metric_train']
        # self._test_split = data['test_split']
        self._train_metadata = data['train_metadata']

        # Callbacks
        if override_callbacks:
            self._print('Overriding model callbacks')

            self._early_stopping = data['early_stopping']
            self._model_checkpoint = data['model_checkpoint']
            self._reduce_lr_on_plateau = data['reduce_lr_on_plateau']

            self._use_tensorboard = data['use_tensorboard']
            self._use_tqdm_notebook = data['use_tqdm_notebook']
            self._use_csv_logger = data['use_csv_logger']
            self._verbose = data['verbose']

        # Update session data
        session_data: Dict[str, Any] = data['session_data']
        for k in session_data.keys():
            self._session_data[k] = session_data[k]

        # Custom session loads
        self._custom_load_session(filename=filename, asserts=False, data=data, check_hash=check_hash)

        self._loaded_session = {
            'description': data['description'],
            'file': filename,
            'last': 'load',
            'train_date': self._train_metadata['train_date']
        }

        self._print(f'Load time:\t{round(time.time() - t0, 3)}s')
        time.sleep(1)
        gc.collect()
        return self

    def update_session(self) -> None:
        """
        Update current loaded session.
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        assert len(self._loaded_session.keys()) == 4, 'Session not saved or loaded'
        self._print('Updating session:')
        self._print(f"\tTarget file: {format(self._loaded_session['file'])}")
        self._print(f"\tUpdate date: {datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')}")

        # Check if model train changed
        update_weights: bool = self._train_metadata['train_date'] != self._loaded_session['train_date']
        if not update_weights:
            self._print('Model weights will not be overwritten')

        self.save_session(
            filename=self._loaded_session['file'],
            description=self._loaded_session['description'],
            save_weights=update_weights
        )
        self._loaded_session['last'] = 'update'

    def load_from_files(self, history_csv_file: str, weights_file: str) -> None:
        """
        Setup model from files (history+weights).

        :param history_csv_file: History files
        :param weights_file: Weights file
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        self.reset_train()
        self._load_weights_from_file(weights_file)
        self._is_trained = True
        self._load_history_from_csv(history_csv_file)

        # Compute metrics
        self._train_redirect = _TRAIN_REDIRECT_METRICS
        self.train(epochs=0, batch_size=0, val_split=0)
        self._train_redirect = _TRAIN_REDIRECT_NONE
        self._extend_train_metadata()
