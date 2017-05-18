"""Builds the Unlimited Hand - sensor values network. (made from MNIST)

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import math
import tensorflow as tf

def inference(sensor_values, layer_units_array):
    """Build the Unlimited Hand - sensor values model up to where it may be used for inference.

    Args:
        sensor_values: sensor values placeholder, from inputs().
        layer_units_array: layer units count array like hidden1, hidden2 and output layer units count array

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    out_layer_units_count = layer_units_array[len(layer_units_array) - 1]
    values = sensor_values
    logits = None

    for layer_index in xrange(len(layer_units_array) - 1):
        name = None
        if layer_index != (len(layer_units_array) - 2):
            name = 'hidden' + str(layer_index + 1)
        else:
            name = 'softmax_linear'

        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([layer_units_array[layer_index], layer_units_array[layer_index + 1]],
                                    stddev=1.0 / math.sqrt(float(layer_units_array[layer_index]))),
                name='weights')
            biases = tf.Variable(tf.zeros([layer_units_array[layer_index + 1]]), name='biases')

            if layer_index != (len(layer_units_array) - 2):
                values = tf.nn.relu(tf.matmul(values, weights) + biases)
            else:
                logits = tf.matmul(values, weights) + biases

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, class_count].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, class_count].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, class_count).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32), name="eval_correct")
