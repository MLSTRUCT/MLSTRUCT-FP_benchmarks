"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - CALLBACKS - TENSORBOARD V2

Tensorboard callback.
"""

__all__ = ['TensorBoardv2']

import os
import six

from keras.callbacks import Callback
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as _k
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


# noinspection PyAttributeOutsideInit,PyProtectedMember,PyCallingNonCallable,PyTypeChecker,PyMissingOrEmptyDocstring,PyMethodMayBeStatic,PyUnresolvedReferences,PyUnusedLocal
class TensorBoardv2(Callback):
    """Enable visualizations for TensorBoard.

    TensorBoard is a visualization tool provided with TensorFlow.

    This callback logs events for TensorBoard, including:

    * Metrics summary plots
    * Training graph visualization
    * Activation histograms
    * Sampled profiling

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:

    ```sh
    tensorboard --logdir=path_to_your_logs
    ```

    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    Arguments:
        log_dir: the path of the directory where to save the log files to be
          parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation and
          weight histograms for the layers of the model. If set to 0, histograms
          won't be computed. Validation data (or split) must be specified for
          histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard. The log file
          can become quite large when write_graph is set to True.
        write_images: whether to write model weights to visualize as image in
          TensorBoard.
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
          writes the losses and metrics to TensorBoard after each batch. The same
          applies for `'epoch'`. If using an integer, let's say `1000`, the
          callback will write the metrics and losses to TensorBoard every 1000
          batches. Note that writing too frequently to TensorBoard can slow down
          your training.
        profile_batch: Profile the batch to sample compute characteristics. By
          default, it will profile the second batch. Set profile_batch=0 to
          disable profiling. Must run in TensorFlow eager mode.
        embeddings_freq: frequency (in epochs) at which embedding layers will
          be visualized. If set to 0, embeddings won't be visualized.
        embeddings_metadata: a dictionary which maps layer name to a file name in
          which metadata for this embedding layer is saved. See the
          [details](
            https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
          about metadata files format. In case if the same metadata file is
          used for all embedding layers, string can be passed.

    Raises:
        ValueError: If histogram_freq is set and no validation data is provided.
    """

    def __init__(
            self,
            log_dir='logs',
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None,
            **kwargs
    ) -> None:
        super().__init__()
        self._validate_kwargs(kwargs)

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        if update_freq == 'batch':
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata

        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._current_batch = 0

        # A collection of file writers currently in use, to be closed when
        # training ends for this callback. Writers are keyed by the
        # directory name under the root logdir: e.g., "train" or
        # "validation".
        self._train_run_name = 'train'
        self._validation_run_name = 'validation'
        self._writers = {}

        self._profile_batch = profile_batch
        # True when a trace is running.
        self._is_tracing = False

    def _validate_kwargs(self, kwargs) -> None:
        """Handle arguments were supported in V1."""
        if kwargs.get('write_grads', False):
            logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                            'for the `TensorBoard` Callback.')
        if kwargs.get('batch_size', False):
            logging.warning('`batch_size` is no longer needed in the '
                            '`TensorBoard` Callback and will be ignored '
                            'in TensorFlow 2.0.')
        if kwargs.get('embeddings_layer_names', False):
            logging.warning('`embeddings_layer_names` is not supported in '
                            'TensorFlow 2.0. Instead, all `Embedding` layers '
                            'will be visualized.')
        if kwargs.get('embeddings_data', False):
            logging.warning('`embeddings_data` is not supported in TensorFlow '
                            '2.0. Instead, all `Embedding` variables will be '
                            'visualized.')

        unrecognized_kwargs = set(kwargs.keys()) - {
            'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
        }

        # Only allow kwargs that were supported in V1.
        if unrecognized_kwargs:
            raise ValueError('Unrecognized arguments in `TensorBoard` '
                             'Callback: ' + str(unrecognized_kwargs))

    def set_model(self, model) -> None:
        """Sets Keras model and writes graph if specified."""
        self.model = model

        # In case this callback is used via native Keras, _get_distribution_strategy does not exist.
        if hasattr(self.model, '_get_distribution_strategy'):
            # TensorBoard callback involves writing a summary file in a
            # possibly distributed settings.
            self._log_write_dir = distributed_file_utils.write_dirpath(
                self.log_dir, self.model._get_distribution_strategy())
        else:
            self._log_write_dir = self.log_dir

        with context.eager_mode():
            self._close_writers()
            if self.write_graph:
                with self._get_writer(self._train_run_name).as_default():
                    with summary_ops_v2.always_record_summaries():

                        if hasattr(model, 'run_eagerly') and not model.run_eagerly:
                            summary_ops_v2.graph(_k.get_graph(), step=0)

                        summary_writable = (
                                self.model._is_graph_network or
                                self.model.__class__.__name__ == 'Sequential')
                        if summary_writable:
                            summary_ops_v2.keras_model('keras', self.model, step=0)

        if self.embeddings_freq:
            self._configure_embeddings()

        summary_state = summary_ops_v2._summary_state
        self._prev_summary_recording = summary_state.is_recording
        self._prev_summary_writer = summary_state.writer
        self._prev_summary_step = summary_state.step

    def _configure_embeddings(self) -> None:
        """Configure the Projector for embeddings."""
        from tensorflow.python.keras.layers import embeddings
        try:
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                              'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for layer in self.model.layers:
            if isinstance(layer, embeddings.Embedding):
                embedding = config.embeddings.add()
                embedding.tensor_name = layer.embeddings.name

                if self.embeddings_metadata is not None:
                    if isinstance(self.embeddings_metadata, str):
                        embedding.metadata_path = self.embeddings_metadata
                    else:
                        if layer.name in embedding.metadata_path:
                            embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

        if self.embeddings_metadata:
            raise ValueError('Unrecognized `Embedding` layer names passed to '
                             '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                             'argument: ' + str(self.embeddings_metadata.keys()))

        # noinspection PyMissingOrEmptyDocstring
        class DummyWriter(object):
            """
            Dummy writer to conform to Projector API.
            """

            def __init__(self, logdir) -> None:
                self.logdir = logdir

            def get_logdir(self) -> str:
                return self.logdir

        writer = DummyWriter(self._log_write_dir)
        projector.visualize_embeddings(writer, config)

    def _close_writers(self) -> None:
        """Close all remaining open file writers owned by this callback.

        If there are no such file writers, this is a no-op.
        """
        with context.eager_mode():
            for writer in six.itervalues(self._writers):
                writer.close()
            self._writers.clear()

    def _get_writer(self, writer_name) -> 'summary_ops_v2.SummaryWriter':
        """Get a summary writer for the given subdirectory under the logdir.

        A writer will be created if it does not yet exist.

        Arguments:
          writer_name: The name of the directory for which to create or
            retrieve a writer. Should be either `self._train_run_name` or
            `self._validation_run_name`.

        Returns:
          A `SummaryWriter` object.
        """
        if writer_name not in self._writers:
            path = os.path.join(self._log_write_dir, writer_name)
            writer = summary_ops_v2.create_file_writer_v2(path)
            self._writers[writer_name] = writer
        return self._writers[writer_name]

    def _set_default_writer(self, writer_name) -> None:
        """Sets the default writer for custom batch-level summaries."""
        if self.update_freq == 'epoch':
            # Writer is only used for custom summaries, which are written
            # batch-by-batch.
            return

        step = self._total_batches_seen[writer_name]

        def _should_record() -> bool:
            return math_ops.equal(step % self.update_freq, 0)

        summary_state = summary_ops_v2._summary_state
        summary_state.is_recording = _should_record
        summary_state.writer = self._get_writer(writer_name)
        summary_ops_v2.set_step(step)

    def _init_batch_steps(self) -> None:
        """Create the total batch counters."""
        if ops.executing_eagerly_outside_functions():
            # Variables are needed for the `step` value of custom tf.summaries
            # to be updated inside a tf.function.
            self._total_batches_seen = {
                self._train_run_name: variables.Variable(0, dtype='int64'),
                self._validation_run_name: variables.Variable(0, dtype='int64')
            }
        else:
            # Custom tf.summaries are not supported in legacy graph mode.
            self._total_batches_seen = {
                self._train_run_name: 0,
                self._validation_run_name: 0
            }

    def _increment_step(self, writer_name) -> None:
        step = self._total_batches_seen[writer_name]
        if isinstance(step, variables.Variable):
            step.assign_add(1)
        else:
            self._total_batches_seen[writer_name] += 1

    def on_train_begin(self, logs=None) -> None:
        self._init_batch_steps()
        if self._profile_batch == 1:
            summary_ops_v2.trace_on(profiler=True)
            self._is_tracing = True

    def on_test_begin(self, logs=None) -> None:
        self._set_default_writer(self._validation_run_name)

    def on_train_batch_end(self, batch, logs=None) -> None:
        """Writes scalar summaries for metrics on every training batch.

        Performs profiling if current batch is in profiler_batches.

        Arguments:
          batch: Integer, index of batch within the current epoch.
          logs: Dict. Metric results for this batch.
        """
        if self.update_freq == 'epoch' and self._profile_batch is None:
            return

        # Don't output batch_size and batch number than TensorBoard summaries
        logs = logs or {}
        train_batches = self._total_batches_seen[self._train_run_name]
        if self.update_freq != 'epoch' and batch % self.update_freq == 0:
            self._log_metrics(logs, prefix='batch_', step=train_batches)

        self._increment_step(self._train_run_name)

        if context.executing_eagerly():
            if self._is_tracing:
                self._log_trace()
            elif (not self._is_tracing and
                  math_ops.equal(train_batches, self._profile_batch - 1)):
                self._enable_trace()

    def on_test_batch_end(self, batch, logs=None) -> None:
        if self.update_freq == 'epoch':
            return
        self._increment_step(self._validation_run_name)

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._set_default_writer(self._train_run_name)

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Runs metrics and histogram summaries at epoch end."""
        self._log_metrics(logs, prefix='epoch_', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def on_train_end(self, logs=None) -> None:
        if self._is_tracing:
            self._log_trace()
        self._close_writers()

        summary_state = summary_ops_v2._summary_state
        summary_state.is_recording = self._prev_summary_recording
        summary_state.writer = self._prev_summary_writer
        summary_state.step = self._prev_summary_step

        # Safely remove the unneeded temp files.
        # In case this callback is used via native Keras, _get_distribution_strategy does not exist.
        if hasattr(self.model, '_get_distribution_strategy'):
            # Safely remove the unneeded temp files.
            distributed_file_utils.remove_temp_dirpath(
                self.log_dir, self.model._get_distribution_strategy())

    def _enable_trace(self) -> None:
        if context.executing_eagerly():
            summary_ops_v2.trace_on(profiler=True)
            self._is_tracing = True

    def _log_trace(self) -> None:
        """Logs the trace graph to TensorBoard."""
        if context.executing_eagerly():
            with self._get_writer(self._train_run_name).as_default(), \
                    summary_ops_v2.always_record_summaries():
                step = _k.get_value(self._total_batches_seen[self._train_run_name])
                summary_ops_v2.trace_export(
                    name='batch_%d' % step,
                    step=step,
                    profiler_outdir=os.path.join(self._log_write_dir, 'train'))
            self._is_tracing = False

    def _log_metrics(self, logs, prefix, step) -> None:
        """Writes metrics out as custom scalar summaries.

        Arguments:
            logs: Dict. Keys are scalar summary names, values are NumPy scalars.
            prefix: String. The prefix to apply to the scalar summary names.
            step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
            self._train_run_name: [],
            self._validation_run_name: [],
        }
        validation_prefix = 'val_'
        for (name, value) in logs.items():
            if name in ('batch', 'size', 'num_steps'):
                # Scrub non-metric items.
                continue
            if name.startswith(validation_prefix):
                name = name[len(validation_prefix):]
                writer_name = self._validation_run_name
            else:
                writer_name = self._train_run_name
            name += prefix  # assign batch or epoch prefix
            logs_by_writer[writer_name].append((name, value))

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Don't create a "validation" events file if we don't
                        # actually have any validation data.
                        continue
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            summary_ops_v2.scalar(name, value, step=step)

    def _log_weights(self, epoch) -> None:
        """Logs the weights of the Model to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), \
                writer.as_default(), \
                summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(':', '_')
                    with ops.init_scope():
                        weight = _k.get_value(weight)
                    summary_ops_v2.histogram(weight_name, weight, step=epoch)
                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, epoch)
            writer.flush()

    def _log_weight_as_image(self, weight, weight_name, epoch) -> None:
        """Logs a weight as a TensorBoard image."""
        w_img = array_ops.squeeze(weight)
        shape = _k.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = _k.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if _k.image_data_format() == 'channels_last':
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = _k.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

        shape = _k.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            summary_ops_v2.image(weight_name, w_img, step=epoch)

    def _log_embeddings(self, epoch) -> None:
        embeddings_ckpt = os.path.join(self._log_write_dir, 'train',
                                       f'keras_embedding.ckpt-{epoch}')
        self.model.save_weights(embeddings_ckpt)
