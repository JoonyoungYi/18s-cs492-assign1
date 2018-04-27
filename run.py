import os
import shutil
import time
import traceback
import random

import tensorflow as tf
import numpy as np

from config import *
from model import fc_model_fn

# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def _get_flipped_matrix(matrix):
    # idx = random.randint(0, matrix.shape[0] - 1)
    # __debug_save_image(matrix[idx, :INPUT_LAYER_SIZE], 'original')
    i_matrix = np.transpose(matrix[:, :INPUT_LAYER_SIZE])
    i_matrix = i_matrix.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
    i_matrix = np.flip(i_matrix, axis=1)
    i_matrix = i_matrix.reshape(INPUT_LAYER_SIZE, -1)
    i_matrix = np.transpose(i_matrix)
    f_matrix = np.concatenate((i_matrix, matrix[:, INPUT_LAYER_SIZE:]), axis=1)
    # __debug_save_image(f_matrix[idx, :INPUT_LAYER_SIZE], 'new')
    return f_matrix


def _flip_train_data(matrix):
    return np.concatenate((matrix, _get_flipped_matrix(matrix[:, :])))


def _get_bolded_matrix(matrix, mask=2):
    file_path = 'data/train-bold-{}.npy'.format(mask)
    if os.path.exists(file_path):
        return np.load(file_path)

    # idx = random.randint(0, matrix.shape[0] - 1)
    # __debug_save_image(matrix[idx, :INPUT_LAYER_SIZE], 'original')
    i_matrix = np.transpose(matrix[:, :INPUT_LAYER_SIZE])
    i_matrix = i_matrix.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)

    row, col, num = i_matrix.shape
    i_matrix_ = np.zeros(i_matrix.shape).astype(np.float32)
    for k in range(num):
        for i in range(row):
            for j in range(col):
                val = np.max(i_matrix[i:min(i + mask, row), j:min(
                    j + mask, col), k:min(k + mask, num)])
                i_matrix_[i, j, k] = val

    i_matrix = i_matrix_.reshape(INPUT_LAYER_SIZE, -1)
    i_matrix = np.transpose(i_matrix)
    b_matrix = np.concatenate((i_matrix, matrix[:, INPUT_LAYER_SIZE:]), axis=1)
    # __debug_save_image(b_matrix[idx, :INPUT_LAYER_SIZE], 'new')
    np.save(file_path, b_matrix)
    return b_matrix


def _bold_train_data(_matrix):
    matrix = _matrix
    # mask 3되면 코드 고쳐야 할듯...
    matrix = np.concatenate((matrix, _get_bolded_matrix(_matrix[:, :],
                                                        mask=2)))
    return matrix


def _augment_train_data(matrix):
    # The order matters because of caching in _bold_train_data function.
    matrix = _bold_train_data(matrix)
    matrix = _rotation_train_data(matrix)
    matrix = _flip_train_data(matrix)
    return matrix
    # return _matrix


def _add_gaussian_noise(train_x):
    # idx = random.randint(0, train_x.shape[0] - 1)
    # __debug_save_image(train_x[idx, :], 'original')
    # batch_x = np.minimum(
    #     np.ones(train_x.shape),
    #     np.maximum(
    #         np.zeros(train_x.shape),
    #         np.add(train_x, np.random.normal(
    #             0, NOISE_STD, train_x.shape)))).astype(np.float32)
    batch_x = np.add(train_x, np.random.normal(
        0, NOISE_STD, train_x.shape)).astype(np.float32)
    # __debug_save_image(batch_x[idx, :], 'new')
    return batch_x


def _get_eval_set():
    eval_set = np.load('data/valid.npy')
    valid_x = eval_set[:, :INPUT_LAYER_SIZE]
    valid_y = eval_set[:, INPUT_LAYER_SIZE].astype(np.int32)
    return valid_x, valid_y


def _get_last_time_msg():
    delta = time.time() - timestamp
    return '(%02d:%02d:%02d)' % (delta // 3600, (delta // 60) % 60, delta % 60)


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


def main(model_idx, refresh=False):
    global timestamp

    print('START MODEL:{}'.format(model_idx))

    if refresh:
        if os.path.exists(MODEL_FOLDER_NAME.format(model_idx)):
            shutil.rmtree(MODEL_FOLDER_NAME.format(model_idx))

    # init dataset and models
    train_set = _augment_train_data(np.load('data/train.npy'))
    test_data = np.load('data/test.npy')
    # print(train_set.shape)
    # assert False

    # setting classifier
    classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir=MODEL_FOLDER_NAME.format(model_idx))

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
    timestamp = time.time()
    for i in range(1, TRAIN_ITER_NUMBER + 1):
        if i % BATCH_ITER_NUMBER == 1:
            # After 1 epoch. reinitilaize batch_x
            e = (i // BATCH_ITER_NUMBER)
            print('EPOCH', e + 1)
            batch_x = _add_gaussian_noise(train_x)
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

    # ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    # ----------------------------------------- ##

    np.save('20183453_network_{}.npy'.format(hidden_layer_number + 1), result)


if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # 7: Layer-3
    # 8: Layer-5
    # 9: Layer-7
    model_idx = 9
    main(model_idx)
