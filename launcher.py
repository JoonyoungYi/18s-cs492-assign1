import os
import shutil
import time
import traceback
import random

import tensorflow as tf
import numpy as np

from config import *
from model import fc_model_fn

tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.path.exists(MODEL_FOLDER_NAME):
    shutil.rmtree(MODEL_FOLDER_NAME)

pa1_dataset = None
timestamp = None


def __debug_save_image(array, name):
    from PIL import Image

    img = Image.new('RGB', (28, 28))
    img.putdata([(int(i * 255), int(i * 255), int(i * 255)) for i in array])
    img.save('{}.png'.format(name))


def _get_rotated_matrix(matrix, k):
    # i_matrix : image matrix, r_matrix : rotated matrix
    assert k >= 1 and k < 4
    i_matrix = np.transpose(matrix[:, :INPUT_LAYER_SIZE])
    i_matrix = i_matrix.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
    i_matrix = np.rot90(i_matrix, k=k)
    i_matrix = i_matrix.reshape(INPUT_LAYER_SIZE, -1)
    i_matrix = np.transpose(i_matrix)
    r_matrix = np.concatenate((i_matrix, matrix[:, INPUT_LAYER_SIZE:]), axis=1)
    return r_matrix


def _rotation_train_data(_matrix):
    matrix = _matrix[:, :]
    for k in range(1, 4):
        matrix = np.concatenate((matrix, _get_rotated_matrix(_matrix, k=k)))
    return matrix


def _flip_train_data(matrix):
    return np.concatenate(
        (np.flip(matrix[:, :INPUT_LAYER_SIZE], axis=1),
         matrix[:, INPUT_LAYER_SIZE:]),
        axis=1)


def _augment_train_data(_matrix):
    matrix = _rotation_train_data(_matrix)
    matrix = np.concatenate((matrix, _flip_train_data(matrix)))
    return matrix


def _add_gaussian_noise(train_x, e):
    # idx = random.randint(0, train_x.shape[0] - 1)
    # __debug_save_image(train_x[idx, :], 'original')
    batch_x = np.minimum(
        np.ones(train_x.shape),
        np.maximum(
            np.zeros(train_x.shape),
            np.add(train_x,
                   np.random.normal(0, min(e * 0.05, 0.2),
                                    train_x.shape)))).astype(np.float32)
    # __debug_save_image(batch_x[idx, :], 'new')
    return batch_x


def _get_eval_set():
    eval_set = np.load('PA1-data/valid.npy')
    valid_x = eval_set[:, :INPUT_LAYER_SIZE]
    valid_y = eval_set[:, INPUT_LAYER_SIZE].astype(np.int32)
    return valid_x, valid_y


def _get_last_time_msg():
    return '(%5.1fsec)' % ((time.time() - timestamp))


def _get_msg_from_result(result):
    return 'acc={}% loss={}'.format(('%3.2f' % (result['accuracy'] * 100))[:4],
                                    '%.4f' % result['loss'])


def _log(i, train_results, eval_results, final=False):
    if final:
        msg = '\nTEST %4s' % ("{:,}".format(i))
    else:
        msg = 'ITER %4s' % ("{:,}".format(i))
    msg += _get_last_time_msg()
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(train_results)
    msg += ', VALID: '
    msg += _get_msg_from_result(eval_results)
    # msg += ', TEST: '
    # msg += _get_test_result_msg(sess, models)
    print(msg)


if __name__ == '__main__':
    # init dataset and models
    train_set = _augment_train_data(np.load('PA1-data/train.npy'))
    timestamp = time.time()

    # setting classifier
    classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir=MODEL_FOLDER_NAME)

    train_x = train_set[:, :INPUT_LAYER_SIZE]
    train_y = train_set[:, INPUT_LAYER_SIZE].astype(np.int32)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=train_set.shape[0],
        num_epochs=1,
        shuffle=False)

    eval_x, eval_y = _get_eval_set()
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_x},
        y=eval_y,
        num_epochs=1,
        shuffle=False, )

    e = 0
    for i in range(1, TRAIN_ITER_NUMBER + 1):
        if i % BATCH_ITER_NUMBER == 1:
            # After 1 epoch. reinitilaize batch_x
            e = (i // BATCH_ITER_NUMBER)
            print('EPOCH', e + 1)
            batch_x = _add_gaussian_noise(train_x, e)
            # batch_x = train_x
            batch_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": batch_x},
                y=train_y,
                batch_size=BATCH_SIZE,
                num_epochs=1,
                shuffle=True)

        classifier.train(input_fn=batch_input_fn, steps=BATCH_SIZE)
        train_results = classifier.evaluate(input_fn=train_input_fn)

        # Eval the model. You can evaluate your trained model with validation data
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        _log(i, train_results, eval_results)

        # Early stop
        if train_results['loss'] < EARLY_STOP_TRAIN_LOSS:
            print('EARLY STOP!')
            break
