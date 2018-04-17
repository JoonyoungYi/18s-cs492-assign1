import os
import shutil
import time

import tensorflow as tf
import numpy as np

from config import *
from model import init_models

mnist = None
train_data, train_labels = None, None
timestamp = None
start_timestamp = time.time()


def _init_dataset():
    # Download MNIST dataset
    global mnist
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # dataset_train = np.load('PA1-data/train.npy')
    # global train_data, train_labels
    # train_data = dataset_train[:, :INPUT_LAYER_SIZE]
    # train_labels = dataset_train[:, INPUT_LAYER_SIZE].astype(np.int32)


def _get_batch_set():
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # batch_x, batch_y = train_data, train_labels
    return batch_x, batch_y


def _get_validation_set():
    valid_x = mnist.validation.images
    valid_y = mnist.validation.labels
    return valid_x, valid_y


def _get_test_set():
    x = mnist.test.images
    y = mnist.test.labels
    return x, y


def _get_validation_result_msg(sess, models, x, y):
    result = sess.run({
        'loss': models['loss'],
        'acc': models['acc']
    }, {
        models['x']: x,
        models['y']: y,
    })
    msg = _get_msg_from_result(result)
    return msg


def _get_msg_from_result(result):
    return 'acc=%.3f loss=%.3f' % (result['acc'], result['loss'])


def _logging_hook(sess, i, result, models):
    global timestamp

    msg = 'STEP %6d' % i
    msg += '(%.2fsec)' % ((time.time() - timestamp))
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(result)

    msg += ', VALID: '
    valid_x, valid_y = _get_validation_set()
    msg += _get_validation_result_msg(sess, models, valid_x, valid_y)
    print(msg)

    timestamp = time.time()


if __name__ == '__main__':
    # init dataset and models
    _init_dataset()
    models = init_models()
    timestamp = time.time()

    # Checkpoint
    saver = tf.train.Saver()

    # Training iteration
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, iter_num + 1):
        batch_x, batch_y = _get_batch_set()

        result = sess.run({
            'loss': models['loss'],
            'acc': models['acc'],
            'train': models['train_op'],
        }, {
            models['x']: batch_x,
            models['y']: batch_y,
        })

        # Validation
        if i % every_n_iter == 0:
            _logging_hook(sess, i, result, models)
            saver.save(sess, './logs/ckpt/model-%d.ckpt' % i)

        if result['loss'] < early_stop_loss:
            break

    test_x, test_y = _get_test_set()
    msg = '\nTEST %6d' % i
    msg += '(%.2fsec): ' % ((time.time() - start_timestamp))
    msg += _get_validation_result_msg(sess, models, test_x, test_y)
    print(msg)
