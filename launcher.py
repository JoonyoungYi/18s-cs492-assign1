import os
import shutil
import time

import tensorflow as tf
import numpy as np

from config import *
from model import init_models

mnist = None
pa1_dataset = None
timestamp = None

# QUESTION: rotate 90도씩 해서 트레이닝에 활용하고 있는데, 이렇게 해도 괜찮은지?


def __debug_save_image(array, name):
    from PIL import Image

    img = Image.new('RGB', (28, 28))
    img.putdata([(int(i * 255), int(i * 255), int(i * 255)) for i in array])
    img.save('{}.png'.format(name))


def _get_rotated_matrix(matrix, k):
    assert k >= 1 and k < 4
    i_matrix = np.transpose(matrix[:, :INPUT_LAYER_SIZE])
    i_matrix = i_matrix.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
    i_matrix = np.rot90(i_matrix, k=k)
    i_matrix = i_matrix.reshape(INPUT_LAYER_SIZE, -1)
    i_matrix = np.transpose(i_matrix)
    r_matrix = np.concatenate((i_matrix, matrix[:, INPUT_LAYER_SIZE:]), axis=1)
    # __debug_save_image(matrix[1, :INPUT_LAYER_SIZE], 'original')
    # __debug_save_image(r_matrix[1, :INPUT_LAYER_SIZE], 'new')
    return r_matrix


def _rotation_train_data(_matrix):
    # i_matrix : image matrix, r_matrix : rotated matrix
    matrix = _matrix[:, :]
    for k in range(3, 4):
        matrix = np.concatenate((matrix, _get_rotated_matrix(_matrix, k=k)))
    return matrix


def _init_dataset():
    global mnist
    global pa1_dataset

    # Download MNIST dataset
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    pa1_dataset = {
        'train': _rotation_train_data(np.load('PA1-data/train.npy')),
        'valid': np.load('PA1-data/valid.npy'),
        'test': np.load('PA1-data/test.npy')
    }


def _get_batch_set(i):
    # batch_x, batch_y = mnist.train.next_batch(batch_size)

    train = pa1_dataset['train']
    idx = np.random.randint(train.shape[0], size=batch_size)
    batch = train[idx, :]
    batch_x = batch[:, :INPUT_LAYER_SIZE]
    batch_y = batch[:, INPUT_LAYER_SIZE]
    return batch_x, batch_y


def _get_validation_set():
    # valid_x = mnist.validation.images
    # valid_y = mnist.validation.labels

    valid_x = pa1_dataset['valid'][:, :INPUT_LAYER_SIZE]
    valid_y = pa1_dataset['valid'][:, INPUT_LAYER_SIZE].astype(np.int32)
    return valid_x, valid_y


def _get_test_set():
    # x = mnist.test.images
    # y = mnist.test.labels
    # return x, y

    valid_x = pa1_dataset['valid'][:, :INPUT_LAYER_SIZE]
    valid_y = pa1_dataset['valid'][:, INPUT_LAYER_SIZE].astype(np.int32)
    return valid_x, valid_y


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
    return 'acc=%2.2f%% loss=%.4f' % (result['acc'] * 100, result['loss'])


def _logging_hook(sess, i, result, models):
    msg = '\STEP %8s' % ("{:,}".format(i))
    msg += '(%.2fsec)' % ((time.time() - timestamp))
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(result)

    msg += ', VALID: '
    valid_x, valid_y = _get_validation_set()
    msg += _get_validation_result_msg(sess, models, valid_x, valid_y)
    print(msg)


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

    result = None
    for i in range(1, iter_num + 1):
        batch_x, batch_y = _get_batch_set(i)

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
    msg = '\nTEST %8s' % ("{:,}".format(i))
    msg += '(%.2fsec): ' % ((time.time() - timestamp))
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(result)
    msg += ' - VALID(or TEST): '
    msg += _get_validation_result_msg(sess, models, test_x, test_y)
    print(msg)
