import tensorflow as tf

from config import *

# def _maxout_layer(x, k=3, idx=0):
#     with tf.device('/gpu:2'):
#         with tf.variable_scope('maxout' + str(idx)):
#             outputs = []
#             for i in range(k):
#                 name = 'w%d_%d' % (idx, i)
#                 w = tf.get_variable(
#                     name,
#                     x.get_shape()[-1],
#                     initializer=tf.constant_initializer(1.0 * (i - k / 2)))
#                 name = 'b%d_%d' % (idx, i)
#                 b = tf.get_variable(
#                     name,
#                     x.get_shape()[-1],
#                     initializer=tf.constant_initializer(i - k / 2))
#                 output = x * w + b
#                 outputs.append(output)
#
#             maxout_layer = tf.reduce_max(outputs, 0)
#             return maxout_layer


def fc_model_fn(features, labels, mode):
    """
        Model function for PA1. Fully Connected(FC) Neural Network.
    """
    # Input Layer
    # I use 1 x 784 flat vector.
    input_layer = features["x"]
    for i in range(hidden_layer_number):
        dense_layer = tf.layers.dense(
            inputs=input_layer,
            units=hidden_layer_size,
            activation=None,
            use_bias=False)
        bn_layer = tf.layers.batch_normalization(
            dense_layer,
            momentum=BN_MOMENTUM,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        activation_layer = tf.nn.relu(bn_layer)
        # activation_layer = tf.nn.leaky_relu(bn_layer, alpha=0.1)
        # activation_layer = _maxout_layer(bn_layer, idx=i)
        dropout_layer = tf.layers.dropout(
            inputs=activation_layer,
            rate=DROPOUT_RATE,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        input_layer = dropout_layer
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
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # This two lines are for bm.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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
