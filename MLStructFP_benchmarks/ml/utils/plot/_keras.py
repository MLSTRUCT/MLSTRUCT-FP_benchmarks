"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - KERAS

Keras utils for plotting.
"""

__all__ = ['plot_model_architecture']

from MLStructFP.utils import DEFAULT_PLOT_DPI

from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers.wrappers import Wrapper

from IPython import display
from tensorflow.python.util import nest
from typing import Optional
import os
import random

# `pydot` is an optional dependency,
# see `extras_require` in `setup.py`.
try:
    import pydot
except ImportError:
    pydot = None


# noinspection PyUnresolvedReferences
def _check_pydot() -> None:
    """Raise errors if `pydot` or GraphViz unavailable."""
    if pydot is None:
        raise ImportError(
            'Failed to import `pydot`. '
            'Please install `pydot`. '
            'For example with `pip install pydot`.')
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError(
            '`pydot` failed to call GraphViz. '
            'Please install GraphViz (https://www.graphviz.org/) '
            'and ensure that its executables are in the $PATH.')


def _is_model(layer) -> bool:
    """
    Check is layer is a model.

    :param layer: Layer object
    :return: True if object is a Model
    """
    return isinstance(layer, Model)


def _is_wrapped_model(layer) -> bool:
    """
    Check if model wrapped.

    :param layer: Layer object
    :return: True if model is a wrapped
    """
    return isinstance(layer, Wrapper) and isinstance(layer.layer, Model)


# noinspection PyUnresolvedReferences
def _add_edge(dot, src, dst) -> None:
    """
    Add edge from two elements.

    :param dot: Dot object
    :param src: Source
    :param dst: Destination
    """
    if not dot.get_edge(src, dst):
        dot.add_edge(pydot.Edge(src, dst))


# noinspection PyProtectedMember,PyUnresolvedReferences
def _model_to_dot_v2(
        model,
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=DEFAULT_PLOT_DPI,
        subgraph=False
) -> 'pydot.Cluster':
    """
    Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.
        subgraph: whether to return a pydot.Cluster instance.

    # Returns
        A `pydot.Dot` instance representing the Keras model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.
    """
    _check_pydot()
    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')

    sub_n_first_node = {}
    sub_n_last_node = {}
    sub_w_first_node = {}
    sub_w_last_node = {}

    if not model._is_graph_network:
        node = pydot.Node(str(id(model)), label=model.name)
        dot.add_node(node)
        return dot
    elif isinstance(model, Sequential):
        if not model.built:
            model.build()
    layers = model._layers

    # Create graph nodes.
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__

        if isinstance(layer, Wrapper):
            if expand_nested and isinstance(layer.layer, Model):
                submodel_wrapper = _model_to_dot_v2(layer.layer, show_shapes,
                                                    show_layer_names, rankdir,
                                                    expand_nested,
                                                    subgraph=True)
                # sub_w : submodel_wrapper
                sub_w_nodes = submodel_wrapper.get_nodes()
                sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
                sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
                dot.add_subgraph(submodel_wrapper)
            else:
                layer_name = f'{layer_name}({layer.layer.name})'
                child_class_name = layer.layer.__class__.__name__
                class_name = f'{class_name}({child_class_name})'

        if expand_nested and isinstance(layer, Model):
            submodel_not_wrapper = _model_to_dot_v2(layer, show_shapes,
                                                    show_layer_names, rankdir,
                                                    expand_nested,
                                                    subgraph=True)
            # sub_n : submodel_not_wrapper
            sub_n_nodes = submodel_not_wrapper.get_nodes()
            sub_n_first_node[layer.name] = sub_n_nodes[0]
            sub_n_last_node[layer.name] = sub_n_nodes[-1]
            dot.add_subgraph(submodel_not_wrapper)

        # Create node's label.
        if show_layer_names:
            label = f'{layer_name}: {class_name}'
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:

            def format_shape(shape) -> str:
                """
                Format shape.
                """
                return str(shape).replace(str(None), '?')

            try:
                outputlabels = format_shape(layer.output_shape)
            except AttributeError:
                outputlabels = '?'
            if hasattr(layer, 'input_shape'):
                inputlabels = format_shape(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [format_shape(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = '?'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)

        if not expand_nested or not isinstance(layer, Model):
            node = pydot.Node(layer_id, label=label)
            dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in nest.flatten(node.inbound_layers):
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        _add_edge(dot, inbound_layer_id, layer_id)
                    else:
                        # if inbound_layer is not Model or wrapped Model
                        if (not isinstance(inbound_layer, Model) and
                                not _is_wrapped_model(inbound_layer)):
                            # if current layer is not Model or wrapped Model
                            if (not isinstance(layer, Model) and
                                    not _is_wrapped_model(layer)):
                                assert dot.get_node(inbound_layer_id)
                                assert dot.get_node(layer_id)
                                _add_edge(dot, inbound_layer_id, layer_id)
                            # if current layer is Model
                            elif isinstance(layer, Model):
                                _add_edge(dot, inbound_layer_id,
                                          sub_n_first_node[layer.name].get_name())
                            # if current layer is wrapped Model
                            elif _is_wrapped_model(layer):
                                _add_edge(dot, inbound_layer_id, layer_id)
                                name = sub_w_first_node[layer.layer.name].get_name()
                                _add_edge(dot, layer_id, name)
                        # if inbound_layer is Model
                        elif isinstance(inbound_layer, Model):
                            name = sub_n_last_node[inbound_layer.name].get_name()
                            if isinstance(layer, Model):
                                output_name = sub_n_first_node[layer.name].get_name()
                                _add_edge(dot, name, output_name)
                            else:
                                _add_edge(dot, name, layer_id)
                        # if inbound_layer is wrapped Model
                        elif _is_wrapped_model(inbound_layer):
                            inbound_layer_name = inbound_layer.layer.name
                            _add_edge(dot,
                                      sub_w_last_node[inbound_layer_name].get_name(),
                                      layer_id)
    return dot


# noinspection PyProtectedMember,PyUnresolvedReferences
def _model_to_dot_v1(
        model,
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=DEFAULT_PLOT_DPI,
        subgraph=False
) -> 'pydot.Cluster':
    """
    Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.
        subgraph: whether to return a pydot.Cluster instance.

    # Returns
        A `pydot.Dot` instance representing the Keras model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.
    """
    _check_pydot()
    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')

    sub_n_nodes = {}
    sub_n_first_node = {}
    sub_n_last_node = {}
    sub_w_nodes = {}
    sub_w_first_node = {}
    sub_w_last_node = {}

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
    layers = model._layers

    # Create graph nodes
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists
        layer_name = layer.name
        class_name = layer.__class__.__name__

        if isinstance(layer, Wrapper):
            if expand_nested and isinstance(layer.layer, Model):
                submodel_wrapper = _model_to_dot_v1(layer.layer, show_shapes,
                                                    show_layer_names, rankdir,
                                                    expand_nested,
                                                    subgraph=True)
                # sub_w : submodel_wrapper
                sub_w_nodes[layer.layer.name] = submodel_wrapper.get_nodes()
                sub_w_first_node[layer.layer.name] = 0
                sub_w_last_node[layer.layer.name] = -1
                dot.add_subgraph(submodel_wrapper)
            else:
                layer_name = f'{layer_name}({layer.layer.name})'
                child_class_name = layer.layer.__class__.__name__
                class_name = f'{class_name}({child_class_name})'

        if expand_nested and isinstance(layer, Model):
            submodel_not_wrapper = _model_to_dot_v1(layer, show_shapes,
                                                    show_layer_names, rankdir,
                                                    expand_nested,
                                                    subgraph=True)
            # sub_n : submodel_not_wrapper
            sub_n_nodes[layer.name] = submodel_not_wrapper.get_nodes()
            sub_n_first_node[layer.name] = 0
            sub_n_last_node[layer.name] = -1
            dot.add_subgraph(submodel_not_wrapper)

        # Create node's label
        if show_layer_names:
            label = f'{layer_name}: {class_name}'
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)

        if not expand_nested or not isinstance(layer, Model):
            node = pydot.Node(layer_id, label=label)
            dot.add_node(node)

    # Connect nodes with edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        _add_edge(dot, inbound_layer_id, layer_id)
                    else:
                        # if inbound_layer is not Model or wrapped Model
                        if not _is_model(inbound_layer) and (
                                not _is_wrapped_model(inbound_layer)):
                            # if current layer is not Model or wrapped Model
                            if not _is_model(layer) and (
                                    not _is_wrapped_model(layer)):
                                assert dot.get_node(inbound_layer_id)
                                assert dot.get_node(layer_id)
                                _add_edge(dot, inbound_layer_id, layer_id)
                            # if current layer is Model
                            elif _is_model(layer):
                                _add_edge(dot, inbound_layer_id,
                                          sub_n_nodes[layer.name][sub_n_first_node[layer.name]].get_name())
                                sub_n_first_node[layer.name] += 1
                            # if current layer is wrapped Model
                            elif _is_wrapped_model(layer):
                                _add_edge(dot, inbound_layer_id, layer_id)
                                name = sub_w_nodes[layer.layer.name][sub_w_first_node[layer.layer.name]].get_name()
                                sub_w_first_node[layer.layer.name] += 1
                                _add_edge(dot, layer_id, name)
                        # if inbound_layer is Model
                        elif _is_model(inbound_layer):
                            name = sub_n_nodes[inbound_layer.name][sub_n_last_node[inbound_layer.name]].get_name()
                            sub_n_last_node[inbound_layer.name] -= 1
                            if _is_model(layer):
                                output_name = sub_n_nodes[layer.name][sub_n_first_node[layer.name]].get_name()
                                sub_n_first_node[layer.name] += 1
                                _add_edge(dot, name, output_name)
                            else:
                                _add_edge(dot, name, layer_id)
                        # if inbound_layer is wrapped Model
                        elif _is_wrapped_model(inbound_layer):
                            inbound_layer_name = inbound_layer.layer.name
                            name = sub_w_nodes[inbound_layer_name][sub_w_last_node[inbound_layer_name]].get_name()
                            sub_w_last_node[inbound_layer_name] -= 1
                            _add_edge(dot, name, layer_id)
    return dot


# noinspection PyUnresolvedReferences
def plot_model_architecture(
        model: 'Model',
        to_file: str = 'model.png',
        show_shapes: bool = False,
        show_layer_names: bool = True,
        rankdir: str = 'TB',
        expand_nested: bool = False,
        dpi: int = 96,
        version: int = 1
) -> Optional['display.Image']:
    """Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.

    # Returns
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    """
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    _remove: bool = False
    if to_file == '':
        to_file = str(random.getrandbits(128)) + '.png'
        _remove = True

    if version == 1:
        dot = _model_to_dot_v1(model, show_shapes, show_layer_names, rankdir,
                               expand_nested, dpi)
    elif version == 2:
        dot = _model_to_dot_v2(model, show_shapes, show_layer_names, rankdir,
                               expand_nested, dpi)
    elif version == 3:
        # noinspection PyArgumentEqualDefault
        dot = plot_model(model, to_file, show_shapes, show_layer_names, rankdir, False, dpi)
        if _remove:
            if os.path.exists(to_file):
                os.remove(to_file)
        return dot
    else:
        raise ValueError('Invalid version')

    # noinspection PyUnresolvedReferences
    dot.write(to_file, format=extension)
    # Returns the image as a Jupyter Image object, to be displayed in-line.
    if extension != 'pdf':
        try:
            dimg = display.Image(filename=to_file)
            if _remove:
                if os.path.exists(to_file):
                    os.remove(to_file)
            return dimg
        except ImportError:
            pass
