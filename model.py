import tensorflow as tf

from config import *


def _init_activation(feature):
    return tf.nn.relu(feature)


def init_models():
    x = tf.placeholder(tf.float32, [None, INPUT_LAYER_SIZE], 'x')
    y = tf.placeholder(tf.int32, [None], 'y')

    hidden_layers = [
        tf.layers.dense(x, hidden_layer_size, activation=_init_activation)
    ]
    for i in range(hidden_layer_number):
        hidden_layers.append(
            tf.layers.dense(
                hidden_layers[-1],
                hidden_layer_size,
                activation=_init_activation))
    output_layer = tf.layers.dense(hidden_layers[-1], OUTPUT_LAYER_SIZE)

    loss = tf.losses.sparse_softmax_cross_entropy(y, output_layer)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5)
    train_op = optimizer.minimize(loss)

    pred = tf.argmax(output_layer, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'train_op': train_op,
        'acc': acc,
        'pred': pred
    }
