import tensorflow as tf
import numpy as np


def iou(true_y, pred_y):
    def f(true_y, pred_y):
        intersection = (true_y * pred_y).sum()
        union = true_y.sum() + pred_y.sum() - intersection
        x = (intersection) / (union)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [true_y, pred_y], tf.float32)
