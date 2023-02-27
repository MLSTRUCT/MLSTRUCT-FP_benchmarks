"""
MLSTRUCTFP BENCHMARKS - ML - UTILS - LOSS

Loss functions.

Sources:
https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
https://www.jeremyjordan.me/semantic-segmentation/
"""

__all__ = [
    'balanced_cross_entropy',
    'binary_cross_entropy',
    'dice_loss',
    'focal_loss',
    'jaccard_distance_loss',
    'pixelwise_softmax_crossentropy',
    'weighted_categorical_crossentropy',
    'weighted_cross_entropy'
]

from keras import backend as k
from tensorflow.python.ops import math_ops
import keras
import tensorflow as tf

from typing import List, Callable


def binary_cross_entropy() -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    :return: Loss function
    """
    print('Loss: Using binary cross entropy')

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss function.
        """
        return keras.losses.binary_crossentropy(y_true, y_pred)

    return loss


def weighted_cross_entropy(beta) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Weighted cross entropy.

    :param beta: Weight factor
    :return: Loss function
    """
    print(f'Loss: Using weighted cross entropy β={beta}')

    def convert_to_logits(y_pred) -> 'tf.Tensor':
        """
        Logits.
        """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss function.
        """
        y_pred = convert_to_logits(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        _loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(_loss)

    return loss


def balanced_cross_entropy(beta) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Balanced cross entropy.

    :param beta: Balance factor
    :return: Loss function
    """
    print(f'Loss: Using balanced cross entropy β={beta}')

    def convert_to_logits(y_pred) -> 'tf.Tensor':
        """
        Logits.
        """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss function.
        """
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        _loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(_loss * (1 - beta))

    return loss


def focal_loss(alpha=0.25, gamma=2) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Focal loss.

    :param alpha:
    :param gamma:
    :return: Loss function
    """
    assert 0 < alpha < 1
    assert 0 < gamma
    print(f'Loss: Using focal loss α={alpha}, γ={gamma}')

    def focal_loss_with_logits(logits, targets, _alpha, _gamma, y_pred) -> 'tf.Tensor':
        """
        Logits.
        """
        weight_a = _alpha * (1 - y_pred) ** _gamma * targets
        weight_b = (1 - _alpha) * y_pred ** _gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                weight_a + weight_b) + logits * weight_b

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss function.
        """
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))
        y_true = math_ops.cast(y_true, y_pred.dtype)

        _loss = focal_loss_with_logits(logits=logits, targets=y_true, _alpha=alpha, _gamma=gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(_loss)

    return loss


def dice_loss(smooth: float = 0) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Dice loss.

    :param smooth: Smooth param
    :return: Loss function
    """
    print('Loss: Using dice loss, smooth={0}', format(smooth))

    def dice_coef(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = k.sum(k.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (k.sum(k.square(y_true), -1) + k.sum(k.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss.
        """
        # noinspection PyTypeChecker
        return -dice_coef(y_true, y_pred)

    return dice_coef_loss


def jaccard_distance_loss(smooth=100) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Jaccard dice loss.

    :param smooth:
    :return: Loss function
    """
    print(f'Loss: Using jaccard dice loss, smooth={smooth}')

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is useful for unbalanced datasets. This has been
        shifted, so it converges on 0 and is smoothed to avoid exploding or disappearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        intersection = k.sum(k.abs(y_true * y_pred), axis=-1)
        sum_ = k.sum(k.abs(y_true) + k.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    return loss


def weighted_categorical_crossentropy(weights: List[float]) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Weighted categorical cross entropy.

    :param weights: Weights
    :return: Loss function
    """
    print(f'Loss: weighhted categorical cross entropy, weights={weights}')

    # weights = [0.9,0.05,0.04,0.01]

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss.
        """
        kweights = k.constant(weights)
        if not k.is_tensor(y_pred):
            y_pred = k.constant(y_pred)
        y_true = k.cast(y_true, y_pred.dtype)
        return k.categorical_crossentropy(y_true, y_pred) * k.sum(y_true * kweights, axis=-1)

    return loss


def pixelwise_softmax_crossentropy(weights: List[float]) -> Callable[['tf.Tensor', 'tf.Tensor'], 'tf.Tensor']:
    """
    Pixelwise weighted softmax cross entropy.

    :param weights: Weights for each loss
    :return: Loss function
    """

    def loss(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        """
        Loss.
        """
        # epsilon = 10e-8
        # output = K.clip(y_pred, epsilon, 1. - epsilon)
        # return -K.sum(y_true * tf.log(output))

        # scale preds so that the class probs of each sample sum to 1
        _EPSILON = 1e-7
        y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        output = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.math.log(output), weights), len(output.get_shape()) - 1)

    return loss
