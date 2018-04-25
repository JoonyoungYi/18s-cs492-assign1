import tensorflow as tf

from config import *


def _init_activation(feature):
    return tf.nn.relu(feature)


def fc_model_fn(features, labels, mode):
    """
        Model function for PA1. Fully Connected(FC) Neural Network.
    """
    # Input Layer
    # I use 1 x 784 flat vector.
    input_layer = features["x"]

    hidden_layers = [
        tf.layers.dense(
            input_layer, hidden_layer_size, activation=_init_activation)
    ]
    for i in range(hidden_layer_number):
        dropout_layer = tf.layers.dropout(
            inputs=hidden_layers[-1],
            rate=DROPOUT_RATE,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        hidden_layer = tf.layers.dense(
            dropout_layer, hidden_layer_size, activation=_init_activation)
        hidden_layers.append(hidden_layer)
    dropout_layer = tf.layers.dropout(
        inputs=hidden_layers[-1],
        rate=DROPOUT_RATE,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    output_layer = tf.layers.dense(dropout_layer, OUTPUT_LAYER_SIZE)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=output_layer, axis=1),
        # Add `softmax_tensor` to the graph.
        # It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    # Also, prepare accuracy.
    loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

    # Setting Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5)
        optimizer = tf.train.AdamOptimizer(learning_rate * 0.1)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={"accuracy": accuracy})
    else:
        # Setting evaluation metrics (for EVAL mode)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={"accuracy": accuracy}, )
