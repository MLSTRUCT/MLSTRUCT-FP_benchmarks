"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - UTILS - KERAS ENGINE - TRAINING ARRAYS

Overwrites keras training arrays methods.
"""

__all__ = ['fit_loop']

from MLStructFP_benchmarks.ml.utils.callbacks import BaseLogger, History

import numpy as np
from keras.engine.training_arrays import cbks, check_num_samples, \
    to_list, should_run_validation, test_loop, K, batch_shuffle, make_batches, slice_arrays
from scipy.sparse import issparse
from typing import TYPE_CHECKING, List, Callable, Union, Optional

if TYPE_CHECKING:
    from keras.models import Model
    from keras.callbacks import Callback


# noinspection PyProtectedMember
def fit_loop(
        model: 'Model',
        model_metrics_names: List[str],
        model_stateful_metrics_names: List[str],
        fit_function: Callable,
        fit_inputs: List,
        out_labels: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        epochs: int = 100,
        verbose: Union[int, bool] = 1,
        callbacks: Optional[List['Callback']] = None,
        val_function: Optional[Callable] = None,
        val_inputs: Optional[List] = None,
        shuffle: bool = True,
        initial_epoch: int = 0,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        validation_freq: int = 1,
        epoch_finish_function: Optional[Callable] = None
) -> 'History':
    """Abstract fit function for `fit_function(fit_inputs)`.

    Assumes that fit_function returns a list, labeled by out_labels.

    # Arguments
        model: Keras model instance.
        model_metrics: Metrics of the model.
        model_stateful_metrics: Stateful metrics of the model.
        fit_function: Keras function returning a list of tensors
        fit_inputs: List of tensors to be fed to `fit_function`
        out_labels: List of strings, display names of
            the outputs of `fit_function`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training and validation
            (if `val_function` and `val_inputs` are not `None`).
        val_function: Keras function to call for validation
        val_inputs: List of tensors to be fed to `val_function`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.
        validation_freq: Only relevant if validation data is provided. Integer
            or list/tuple/set. If an integer, specifies how many training
            epochs to run before a new validation run is performed, e.g.,
            validation_freq=2` runs validation every 2 epochs. If a list,
            tuple, or set, specifies the epochs on which to run validation,
            e.g. `validation_freq=[1, 2, 10]` runs validation at the end
            of the 1st, 2nd, and 10th epochs.
        epoch_finish_function: Function triggered once the epoch has finished

    # Returns
        `History` object.
    """
    do_validation = False
    if val_function and val_inputs:
        do_validation = True
        if (verbose and fit_inputs and
                hasattr(fit_inputs[0], 'shape') and hasattr(val_inputs[0], 'shape')):
            print('Train on %d samples, validate on %d samples' %
                  (fit_inputs[0].shape[0], val_inputs[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')
    elif do_validation:
        if steps_per_epoch:
            raise ValueError('Must specify `validation_steps` '
                             'to perform validation '
                             'when doing step-wise training.')

    num_train_samples = check_num_samples(fit_inputs,
                                          batch_size=batch_size,
                                          steps=steps_per_epoch,
                                          steps_name='steps_per_epoch')

    index_array = None
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

    model.history = History()
    _callbacks = [BaseLogger(stateful_metrics=model_stateful_metrics_names)]
    if verbose:
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        _callbacks.append(
            cbks.ProgbarLogger(count_mode, stateful_metrics=model_stateful_metrics_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)
    out_labels = out_labels or []

    assert len(out_labels) == len(model_metrics_names)

    # It's possible to call back a different model than itself
    # (used by Sequential models)
    callback_model = model._get_callback_model()
    callback_metrics = model_metrics_names.copy()
    if do_validation:
        callback_metrics += ['val_' + n for n in model_metrics_names]

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics
    })
    callbacks._call_begin_hook('train')
    callbacks.model.stop_training = False
    for cbk in callbacks:
        cbk.validation_data = val_inputs

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(fit_inputs[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {'batch': step_index, 'size': 1}
                callbacks._call_batch_hook('train', 'begin', step_index, batch_logs)
                outs = fit_function(fit_inputs)

                outs = to_list(outs)
                for lab, o in zip(out_labels, outs):
                    batch_logs[lab] = o

                callbacks._call_batch_hook('train', 'end', step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation and should_run_validation(validation_freq, epoch):
                val_outs = test_loop(model, val_function, val_inputs,
                                     steps=validation_steps,
                                     callbacks=callbacks)
                val_outs = to_list(val_outs)
                # Same labels assumed
                for lab, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + lab] = o
        else:
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(num_train_samples, batch_size)
            batch_index = 0
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(fit_inputs[-1], int):
                        # Do not slice the training phase flag
                        ins_batch = slice_arrays(
                            fit_inputs[:-1], batch_ids) + [fit_inputs[-1]]
                    else:
                        ins_batch = slice_arrays(fit_inputs, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
                callbacks._call_batch_hook('train', 'begin', batch_index, batch_logs)
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                outs = fit_function(ins_batch)
                outs = to_list(outs)
                for lab, o in zip(out_labels, outs):
                    batch_logs[lab] = o

                callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
                if callbacks.model.stop_training:
                    break

            if batch_index == len(batches) - 1:  # Last batch
                if do_validation and should_run_validation(validation_freq, epoch):
                    val_outs = test_loop(model, val_function, val_inputs,
                                         batch_size=batch_size,
                                         callbacks=callbacks)
                    val_outs = to_list(val_outs)
                    # Same labels assumed
                    for lab, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + lab] = o

        callbacks.on_epoch_end(epoch, epoch_logs)
        if epoch_finish_function is not None:
            epoch_finish_function(epoch)
        if callbacks.model.stop_training:
            break
    callbacks._call_end_hook('train')
    return model.history
