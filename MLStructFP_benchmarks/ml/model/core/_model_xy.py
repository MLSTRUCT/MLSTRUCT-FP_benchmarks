"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - CORE - XY

Model based on xy vectors.
"""

__all__ = ['GenericModelXY']

from abc import ABCMeta

from MLStructFP_benchmarks.ml.model.core import GenericModel, ModelDataXY
from MLStructFP_benchmarks.ml.model.core._model import _ERROR_MODEL_IN_PRODUCTION, _ERROR_MODEL_NOT_IN_PRODUCTION
from MLStructFP_benchmarks.ml.utils import get_key_hash, r2_score

from sklearn import preprocessing
from typing import List, Union, Tuple, Any, Dict, Optional
import copy
import hashlib
import numpy as np
import pandas as pd


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


def _remove_duplicates_from_list(source: List[Any]) -> List[Any]:
    """
    Remove duplicates from list.

    :param source: Source list
    :return: List without duplicates
    """
    newl: List[Any] = []
    for s in source:
        if s not in newl:
            newl.append(s)
    return newl


def _assert_shape(x: 'pd.DataFrame', y: 'pd.DataFrame', msg: str) -> None:
    """
    Assert shapes.

    :param x: X shape (pandas)
    :param y: Y shape (pandas)
    :param msg: Error message
    """
    xx = x.shape
    yy = y.shape
    assert xx[0] == yy[0], 'Number of rows does not match at ' + msg
    # assert xx[1] == yy[1], 'Number of columns does not match at ' + msg


# noinspection PyUnusedLocal
class GenericModelXY(GenericModel, metaclass=ABCMeta):
    """
    XY tensor model.
    """
    _assert_data: bool
    _data: 'ModelDataXY'
    _data_classification_library: Dict[str, List[str]]  # Classification literals
    _ignore_y_cols: List[str]  # Ignore columns in y
    _is_scaled: bool  # Data was scaled on train
    _min_max_scaler: Dict[str, 'preprocessing.MinMaxScaler']
    _x_col_names: List[str]  # Name of columns in
    _x_pd: 'pd.DataFrame'
    _x_pd_id: 'pd.DataFrame'
    _x_test: 'pd.DataFrame'
    _x_test_id: 'pd.DataFrame'
    _x_train: 'pd.DataFrame'
    _x_train_id: 'pd.DataFrame'
    _y_col_names: List[str]  # Name of columns in y
    _y_pd: 'pd.DataFrame'
    _y_pd_id: 'pd.DataFrame'
    _y_test: 'pd.DataFrame'
    _y_test_id: 'pd.DataFrame'
    _y_train: 'pd.DataFrame'
    _y_train_id: 'pd.DataFrame'

    def __init__(self, data: Optional['ModelDataXY'], name: str, *args, **kwargs) -> None:
        """
        Constructor model XY.

        :param data: Model data
        :param name: Model name
        :param sel_columns_x: Select columns on x data, if empty select all
        :param sel_columns_y: Select columns on y data, if empty select all
        :param args: Optional non-keyword arguments
        :param kwargs: Optional keyword arguments
        """
        GenericModel.__init__(self, name=name, path=kwargs.get('path', ''))
        sel_columns_x: Union[str, List[str]] = kwargs.get('sel_columns_x', '')
        sel_columns_y: Union[str, List[str]] = kwargs.get('sel_columns_y', '')

        # Scaler
        self._ignore_y_cols = []
        self._min_max_scaler = {}
        self._is_scaled = False

        if data is None:
            assert isinstance(sel_columns_x, list) and isinstance(sel_columns_y, list), \
                'Selection columns must be lists'
            assert len(sel_columns_x) > 0 and len(sel_columns_y) > 0, 'Selection columns cannot be empty'
            self._x_col_names = _remove_duplicates_from_list(sel_columns_x)
            self._y_col_names = _remove_duplicates_from_list(sel_columns_y)
            self._assert_data = False
            self.ignore_output_column(column=kwargs.get('y_col_ignore', []))
            return

        # Save data
        assert data.__class__.__name__ == 'ModelDataXY', \
            f'Invalid data class <{data.__class__.__name__}>'
        self._data = data
        self._assert_data = False
        self._is_scaled = data.is_scaled()

        # Get train test (all pandas data frames)
        self._x_train, self._x_train_id = \
            self._data.get_dataframe('xtrain', drop=True), self._data.get_dataframe('xtrain', get_id=True)
        self._x_test, self._x_test_id = \
            self._data.get_dataframe('xtest', drop=True), self._data.get_dataframe('xtest', get_id=True)
        self._y_train, self._y_train_id = \
            self._data.get_dataframe('ytrain', drop=True), self._data.get_dataframe('ytrain', get_id=True)
        self._y_test, self._y_test_id = \
            self._data.get_dataframe('ytest', drop=True), self._data.get_dataframe('ytest', get_id=True)
        self._x_pd, self._x_pd_id = \
            self._data.get_dataframe('x', drop=True), self._data.get_dataframe('x', get_id=True)
        self._y_pd, self._y_pd_id = \
            self._data.get_dataframe('y', drop=True), self._data.get_dataframe('y', get_id=True)

        # Assert shapes
        _assert_shape(self._x_train, self._y_train, 'train dataframe')
        _assert_shape(self._x_test, self._y_test, 'test dataframe')
        _assert_shape(self._x_pd, self._y_pd, 'dataframe')

        # Compute test split data
        self._test_split = self._x_test.shape[0] / (self._x_train.shape[0] + self._x_test.shape[0])

        # Check selection columns
        col_x: List[str] = list(self._x_pd.columns)
        col_y: List[str] = list(self._y_pd.columns)

        if sel_columns_x == '':
            sel_columns_x = col_x
        if sel_columns_y == '':
            sel_columns_y = col_y
        if isinstance(sel_columns_x, str):
            sel_columns_x = [sel_columns_x]
        if isinstance(sel_columns_y, str):
            sel_columns_y = [sel_columns_y]
        if not isinstance(sel_columns_x, list):
            sel_columns_x = [sel_columns_x]
        if not isinstance(sel_columns_y, list):
            sel_columns_x = [sel_columns_y]

        # Remove duplicates
        sel_columns_x = _remove_duplicates_from_list(sel_columns_x)
        sel_columns_y = _remove_duplicates_from_list(sel_columns_y)

        # Create columns names
        self._x_col_names: List[str] = []
        self._y_col_names: List[str] = []

        assert len(sel_columns_x) > 0, 'At least 1 column must be selected on x data'
        assert len(sel_columns_y) > 0, 'At least 1 column must be selected on y data'

        # Check selection not in drop
        for sx in sel_columns_x:
            assert sx not in data.get_drop('x'), f'Column <{sx}> selected on x data is in data drop'
            assert sx in col_x, f'Column <{sx}> does not exist in x data'
        for sy in sel_columns_y:
            assert sy not in data.get_drop('y'), f'Column <{sy}> selected on y data is in data drop'
            assert sy in col_y, f'Column <{sy}> does not exist in y data'

        # Filter
        self._x_train = self._x_train[sel_columns_x]
        self._x_test = self._x_test[sel_columns_x]
        self._x_pd = self._x_pd[sel_columns_x]

        self._y_train = self._y_train[sel_columns_y]
        self._y_test = self._y_test[sel_columns_y]
        self._y_pd = self._y_pd[sel_columns_y]

        # Save columns in the same order as data
        for _s in self._x_pd.columns:
            self._x_col_names.append(_s)
        for _s in self._y_pd.columns:
            self._y_col_names.append(_s)

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
        return self._name + f' + {len(self._x_col_names)}->{len(self._y_col_names)}\n' + _layers

    def ignore_output_column(self, column: Union[str, List[str]]) -> None:
        """
        Ignore output columns.

        :param column: Column names
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if column == '':
            return
        if isinstance(column, str):
            column = [column]
        assert isinstance(column, list), 'Column must be a list'
        for k in column:
            assert k in self._y_col_names, f'Column <{k}> cannot be ignored as it does not exists on Y output'
            if k not in self._ignore_y_cols:
                self._ignore_y_cols.append(k)

    def get_ignored_columns(self) -> List[str]:
        """
        :return: Ignored column list
        """
        return self._ignore_y_cols.copy()

    def get_column_index(self, column: str, xy: str = 'xy') -> Tuple[int, int]:
        """
        Returns the columns ID from a given column name.

        :param column: Column name
        :param xy: Which dataframes to get from
        :return: x, y indexes
        """
        if xy == 'xy':
            if column not in self._x_col_names:
                raise ValueError(f'Column <{column}> does not exist in X dataframe')
            if column not in self._y_col_names:
                raise ValueError(f'Column <{column}> does not exist in Y dataframe')
            return self._x_col_names.index(column), self._y_col_names.index(column)
        elif xy == 'x':
            if column not in self._x_col_names:
                raise ValueError(f'Column <{column}> does not exist in X dataframe')
            return self._x_col_names.index(column), -1
        elif xy == 'y':
            if column not in self._y_col_names:
                raise ValueError(f'Column <{column}> does not exist in Y dataframe')
            return self._y_col_names.index(column), -1
        else:
            raise ValueError('xy must be "x", "y" or "xy"')

    def get_column_name(self, xy: str, index: int) -> str:
        """
        Returns the name of the column.

        :param xy: Which column
        :param index: Column index
        :return: Name
        """
        if xy == 'x':
            return self._x_col_names[index]
        elif xy == 'y':
            return self._y_col_names[index]
        else:
            raise ValueError('Invalid xy, only "x" or "y" valid')

    def get_xy(self, xy: str) -> Tuple['pd.DataFrame', 'pd.DataFrame']:
        """
        Returns the xy datasets.

        :param xy: Which dataframe
        :return: Dataframes
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if xy == 'dataset':
            return self._x_pd, self._y_pd
        elif xy == 'train':
            return self._x_train, self._y_train
        elif xy == 'test':
            return self._x_test, self._y_test
        else:
            raise ValueError('invalid xy parameter, valid "dataset", "train" or "test"')

    def get_xy_id_df(self, xy: str) -> Tuple['pd.DataFrame', 'pd.DataFrame']:
        """
        Returns the ID xy datasets.

        :param xy: Which dataframe
        :return: Dataframes
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if xy == 'dataset':
            return self._x_pd_id, self._y_pd_id
        elif xy == 'train':
            return self._x_train_id, self._y_train_id
        elif xy == 'test':
            return self._x_test_id, self._y_test_id
        else:
            raise ValueError('invalid xy parameter, valid "dataset", "train" or "test"')

    def get_index_by_object_id_df(self, xy: str, xory: str, column_id: str, object_id: int) -> int:
        """
        Returns the object index in matrix values by an ID.

        :param xy: Which dataframe
        :param xory: Use X or Y datasets
        :param column_id: Column ID name
        :param object_id: Object ID, must be unique
        :return: Position
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if xory == 'x':
            id_col, _ = self.get_xy_id_df(xy)
        elif xory == 'y':
            _, id_col = self.get_xy_id_df(xy)
        else:
            raise ValueError('xory must be "x" or "y"')

        # Get ID position of the dataframes
        assert column_id in id_col.columns, \
            f"Column ID <{column_id}> does not exists on dataframe <{xory + '+' + xy}>"
        obj_indx: 'pd.DataFrame' = id_col[id_col[column_id] == object_id]
        if len(obj_indx.index.values) == 0:
            raise ValueError(f'ID <{object_id}> does not exist in dataframe <{xy}>')
        elif len(obj_indx.index.values) > 1:
            raise ValueError(f'ID <{object_id}> has multiple values in dataframe <{xy}>')
        indx = int(obj_indx.index.values[0])

        return indx

    def get_id_list(self, xy: str, id_column: str) -> List[int]:
        """
        Returns the ID of each dataset.

        :param xy: Which dataset use
        :param id_column: Which column will be used
        :return: List of unique ids
        """
        if self._production:
            raise RuntimeError(_ERROR_MODEL_IN_PRODUCTION)
        if xy == 'dataset':
            _id_dataset = self._x_pd_id[id_column].unique()
            id_dataset = []
            for i in _id_dataset:
                id_dataset.append(i)
            id_dataset.sort()
            return id_dataset
        elif xy == 'test':
            _id_test = self._x_test_id[id_column].unique()
            id_test = []
            for i in _id_test:
                id_test.append(i)
            id_test.sort()
            return id_test
        elif xy == 'train':
            _id_train = self._x_train_id[id_column].unique()
            id_train = []
            for i in _id_train:
                id_train.append(i)
            id_train.sort()
            return id_train
        else:
            raise ValueError('Invalid xy, "dataset", "train" or "test" are valid')

    def inverse_column_minmax_scaler(self, column_name: str, x: 'np.ndarray') -> 'np.ndarray':
        """
        Inverse row vector from scaler.

        :param column_name: Column name
        :param x: Data column to inverse, it will return also a column
        :return: Inversed data
        """
        if self._production:
            assert column_name in list(self._min_max_scaler.keys()), \
                f'Column <{column_name}> is not defined in the scaler'
            return self._min_max_scaler[column_name].inverse_transform(x.reshape(-1, 1)).reshape(1, -1)[0, :]
        return self._data.inverse_minmax_scaler(column_name=column_name, x=x)

    def _get_column_mask_id(
            self,
            xy: str,
            col: str,
            project_id: Optional[Union[int, List[int]]],
            project_id_column: str
    ) -> Optional['pd.Series']:
        """
        Returns a mask by ID.

        :param xy: Which dataset use
        :param col: Which column use
        :param project_id: ID of the project, if None does not use ID
        :param project_id_column: Name of the project ID column
        :return: Mask or None
        """
        xy_id = self.get_id_list(xy=xy, id_column=project_id_column)
        _x, _y = self.get_xy_id_df(xy=xy)
        _a: 'pd.DataFrame'
        if col == 'x':
            _a = _x
        elif col == 'y':
            _a = _y
        else:
            raise ValueError('Invalid col, use "x" or "y"')
        mask = None
        if project_id is not None:
            if not isinstance(project_id, list):
                project_id = [project_id]
            for _id in project_id:
                assert _id in xy_id, f'Project <{_id}> not found in dataframe <{xy}>'
            mask = _a[project_id_column] == project_id[0]
            # _test_mask(mask)
            for _i in range(1, len(project_id)):
                mask = (_a[project_id_column] == project_id[_i]) | mask
                # _test_mask(mask)
        return mask

    def eval_r2_all(
            self,
            xy: str = 'test',
            use_model: bool = True,
            project_id: Optional[Union[int, List[int]]] = None,
            project_id_column: str = 'projectID',
            data_filter: Optional['pd.Series'] = None
    ) -> Dict[str, float]:
        """
        Evaluate R² score on all output columns.

        :param xy: Which dataset use
        :param use_model: Use model
        :param project_id: ID of the project, if None does not use ID
        :param project_id_column: Name of the project ID column
        :param data_filter: Use data filter
        :return: R² float score dict for each column
        """
        r2 = {}
        for k in self._y_col_names:
            try:
                r2[k] = self.get_r2(
                    column=k,
                    xy=xy,
                    use_model=use_model,
                    project_id=project_id,
                    project_id_column=project_id_column,
                    data_filter=data_filter
                )
            except ValueError:
                print(f'Column <{k}> does not exists')
                r2[k] = -1
        return r2

    def get_real_predicted_column(
            self,
            column: str,
            xy: str = 'test',
            use_model: bool = True,
            project_id: Optional[Union[int, List[int]]] = None,
            project_id_column: str = 'projectID',
            data_filter: Optional['pd.Series'] = None
    ) -> 'pd.DataFrame':
        """
        Returns a pandas comparison between real and predicted column values.

        :param column: Column to compare
        :param xy: Which dataset to use
        :param use_model: Use model or not
        :param project_id: Project ID filter
        :param project_id_column: Column project ID filter
        :param data_filter: Use external data filter
        :return: Dataframe of real/predicted
        """
        pass

    def get_r2(
            self,
            column: str,
            xy: str = 'test',
            use_model: bool = True,
            project_id: Optional[Union[int, List[int]]] = None,
            project_id_column: str = 'projectID',
            data_filter: Optional['pd.Series'] = None
    ) -> float:
        """
        Returns the R² score by project id.

        :param column: Column data name to get R² score
        :param xy: Which dataset use
        :param use_model: Use model
        :param project_id: ID of the project, if None does not use ID
        :param project_id_column: Name of the project ID column
        :param data_filter: Use data filter
        :return: R² float score
        """
        pdf = self.get_real_predicted_column(column, xy, use_model, project_id, project_id_column, data_filter)
        x = pdf['real']
        y = pdf['predicted']
        key = get_key_hash(column, project_id, project_id_column, xy)
        r2 = r2_score(x, y, False)[0]
        if use_model and key != '':
            self._register_train_data('r2_' + key, r2)
        return r2

    def get_data_columns(self, remove_ignored: bool = True) -> Dict[str, List[str]]:
        """
        Returns the data column names from x to y.

        :param remove_ignored: Remove ignored columns from output
        :return: Name dict
        """
        assert isinstance(remove_ignored, bool)
        yc = []
        for y in self._y_col_names:
            if remove_ignored and y in self._ignore_y_cols:
                continue
            yc.append(y)
        return {
            'x': self._x_col_names,
            'y': yc
        }

    def _get_columns_name_hash(self) -> str:
        """
        :return: Returns the columns name hash
        """
        h = hashlib.md5()
        h.update(_hash_list(self._x_col_names).encode())
        h.update(_hash_list(self._y_col_names).encode())
        return h.hexdigest()

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        See upper doc.
        """
        # Columns
        data['xy_col_shape'] = (len(self._x_col_names), len(self._y_col_names))
        data['x_col_names'] = self._x_col_names
        data['y_col_names'] = self._y_col_names
        data['y_col_ignore'] = self._ignore_y_cols

        # Data info
        data['data_filename'] = self._data.get_filename()
        data['data_xy_scaled'] = self._data.is_scaled()

        # Save minmax scaler values
        minmax_scales = {'x': {}, 'y': {}}
        for cx in self._x_col_names:
            sc = self._data.get_minmax_scaler_limits(cx)
            if sc[0] == 0 and sc[1] == 0:
                continue
            minmax_scales['x'][cx] = sc
        for cy in self._y_col_names:
            sc = self._data.get_minmax_scaler_limits(cy)
            if sc[0] == 0 and sc[1] == 0:
                continue
            minmax_scales['y'][cy] = sc
        data['data_xy_scales'] = minmax_scales

        # Save classification library
        data['data_xy_classification_library'] = self._data.get_classification_literals()

        # Save hashes
        data['hash_columns'] = self._get_columns_name_hash()
        data['hash_test'] = self._data.get_data_hash('test')
        data['hash_train'] = self._data.get_data_hash('train')
        data['hash_xy'] = self._data.get_data_hash('xy')

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
        if asserts:
            if check_hash:
                assert data['hash_columns'] == self._get_columns_name_hash(), 'Columns hash changed'
                if self._assert_data:
                    assert data['hash_xy'] == self._data.get_data_hash('xy'), 'Data xy hash changed'
                    assert data['hash_train'] == self._data.get_data_hash('train'), 'Data train hash changed'
                    assert data['hash_test'] == self._data.get_data_hash('test'), 'Data test hash changed'
                    assert data['data_xy_scaled'] == self._is_scaled

        else:
            # Load scaler
            scalex: dict = data['data_xy_scales']['x']
            scaley: dict = data['data_xy_scales']['y']
            for k in scalex.keys():
                scaler = preprocessing.MinMaxScaler()
                scaler.min_, scaler.scale_ = scalex[k][0], scalex[k][1]
                self._min_max_scaler[k] = scaler
            for k in scaley.keys():
                if k in self._min_max_scaler.keys():
                    continue
                scaler = preprocessing.MinMaxScaler()
                scaler.min_, scaler.scale_ = scaley[k][0], scaley[k][1]
                self._min_max_scaler[k] = scaler
            self._is_scaled = data['data_xy_scaled']

            # Ignore columns
            self._ignore_y_cols = []
            self.ignore_output_column(column=data['y_col_ignore'])

            # Load classification
            self._data_classification_library = data['data_xy_classification_library']

    def get_classification_library(self) -> Dict[str, List[str]]:
        """
        :return: Returns the classification library only in production
        """
        if self._production:
            return copy.deepcopy(self._data_classification_library)
        raise RuntimeError(_ERROR_MODEL_NOT_IN_PRODUCTION)

    def make_output_y_df(self, y: 'np.ndarray', inverse: bool = True, remove_ignored: bool = True) -> 'pd.DataFrame':
        """
        Create a dataframe from a output.

        :param y: Output values
        :param inverse: Inverse values from scaler
        :param remove_ignored: Remove ignored columns
        :return: Dataframe object
        """
        assert isinstance(y, np.ndarray), \
            f'y must be a numeric value nparray object, <{y.__class__.__name__}> class was given'
        shp = y.shape
        assert shp[0] > 0, 'y rows cannot be empty'
        assert shp[1] == len(self._y_col_names), 'Number of output columns does not match'
        df = pd.DataFrame(y)
        df.columns = self._y_col_names
        if inverse:
            df = self.inverse_transform_scaler_xy(df)
        if remove_ignored:
            df = df.drop(columns=self._ignore_y_cols)
        return df

    def transform_x(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Transform input for being compatible with model.

        :param data: Data
        :return: Transformed data
        """
        # Check all input columns exists on the dataframe
        for c in self._x_col_names:
            assert c in data.columns, f'Input x column <{c}> does not exists in data'

        # Select columns in order
        x = data[self._x_col_names]

        return x

    def transform_scaler_xy(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Transform data for model prediction.

        :param data: Data to scale
        :return: Scaled data
        """
        if not self._is_scaled:
            return data
        data = data.copy()
        scc = list(self._min_max_scaler.keys())
        for c in data.columns:
            if c in scc:
                data[c] = self._min_max_scaler[c].transform(data[c].values.reshape(-1, 1))
            else:
                raise RuntimeError(f'Column <{c}> is not defined in the scaler')
        return data

    def inverse_transform_scaler_xy(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Inverse transform data from model prediction.

        :param data: Data to scale
        :return: Scaled data
        """
        if not self._is_scaled:
            return data
        data = data.copy()
        scc = list(self._min_max_scaler.keys())
        for c in data.columns:
            if c in scc:
                data[c] = self._min_max_scaler[c].inverse_transform(data[c].values.reshape(-1, 1))
            else:
                raise RuntimeError(f'Column <{c}> is not defined in the scaler')
        return data

    # noinspection PyMethodMayBeStatic
    def join_output_list_as_np(self, y: Union[List['np.ndarray'], 'np.ndarray']) -> 'np.ndarray':
        """
        Join output vector lists as a single array.

        :param y: List of arrays
        :return: Single array
        """
        if isinstance(y, np.ndarray):
            return y
        a = len(y[0])

        # Assert all vectors have the same length, also compute the output shape
        b = 0  # Number of columns
        for _ in y:
            x = y[0].shape
            assert len(x) == 2, 'Only 2 dimensional vector are allowed'
            assert x[0] == a, 'Length of vector changed'
            b += x[1]

        oy = np.zeros((a, b), dtype=y[0].dtype)
        for j in range(len(y)):  # Iterate through each output
            for k in range(y[j].shape[1]):  # Column
                oy[:, j + k] = y[j][:, k]

        return oy
