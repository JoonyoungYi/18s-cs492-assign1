import os
import shutil
import time
import traceback

import tensorflow as tf
import numpy as np

from config import *
from model import init_models
from model import fc_model_fn

tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.path.exists(MODEL_FOLDER_NAME):
    shutil.rmtree(MODEL_FOLDER_NAME)

mnist = None
pa1_dataset = None
timestamp = None


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
    return _matrix


def _init_dataset():
    global mnist
    global pa1_dataset

    # Download MNIST dataset
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    pa1_dataset = {
        'train': _augment_train_data(np.load('PA1-data/train.npy')),
        'valid': np.load('PA1-data/valid.npy'),
        'test': np.load('PA1-data/test.npy')
    }


def _get_batch_set():
    # batch_x, batch_y = mnist.train.next_batch(batch_size)

    train = pa1_dataset['train']
    idx = np.random.randint(train.shape[0], size=BATCH_SIZE)
    batch = train[idx, :]
    batch_x = batch[:, :INPUT_LAYER_SIZE]
    batch_x = np.minimum(
        np.ones(batch_x.shape),
        np.maximum(
            np.zeros(batch_x.shape),
            np.add(batch_x, np.random.normal(0, 0.2, batch_x.shape)))).astype(np.float32)
    batch_y = batch[:, INPUT_LAYER_SIZE].astype(np.int32)
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

    x = pa1_dataset['test'][:, :INPUT_LAYER_SIZE]
    return x, None


def _get_last_time_msg():
    return '(%3.1fsec)' % ((time.time() - timestamp))


def _get_valid_result_msg(sess, models):
    valid_x, valid_y = _get_validation_set()
    result = sess.run({
        'loss': models['loss'],
        'acc': models['acc']
    }, {
        models['x']: valid_x,
        models['y']: valid_y,
    })
    msg = _get_msg_from_result(result)
    return msg


def _get_msg_from_result(result):
    return 'acc={} loss={}'.format(('%2.2f%%' % (result['acc'] * 100))[:5],
                                   '%.4f' % result['loss'])


def _get_test_result_msg(sess, models):
    x, y = _get_test_set()
    result = sess.run({
        'loss': models['loss'],
        'acc': models['acc']
    }, {
        models['x']: x,
        models['y']: y,
    })
    msg = _get_msg_from_result(result)
    return msg


def _logging_hook(sess, saver, i, result, models, final=False):
    if final:
        msg = '\nTEST %8s' % ("{:,}".format(i))
    else:
        msg = 'STEP %8s' % ("{:,}".format(i))
    msg += _get_last_time_msg()
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(result)
    msg += ', VALID: '
    msg += _get_valid_result_msg(sess, models)
    # msg += ', TEST: '
    # msg += _get_test_result_msg(sess, models)
    print(msg)

    saver.save(sess, './logs/ckpt/model-%d.ckpt' % i)


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

    # Save model and checkpoint
    classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir=MODEL_FOLDER_NAME)

    eval_x, eval_y = _get_validation_set()
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_x},
        y=eval_y,
        num_epochs=1,
        shuffle=False, )

    for i in range(1, TRAIN_ITER_NUMBER + 1):
        x, y = _get_batch_set()
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x}, y=y, batch_size=BATCH_SIZE, num_epochs=1, shuffle=True)
        classifier.train(input_fn=train_input_fn, steps=BATCH_SIZE)
        train_results = classifier.evaluate(input_fn=train_input_fn)
        print(train_results)
        print(x.shape, y.shape)
        # Eval the model. You can evaluate your trained model with validation data
        # try:
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    # result = None
    # try:
    #     for i in range(1, iter_num + 1):
    #         batch_x, batch_y = _get_batch_set(i)
    #
    #         result = sess.run({
    #             'loss': models['loss'],
    #             'acc': models['acc'],
    #             'train': models['train_op'],
    #         }, {
    #             models['x']: batch_x,
    #             models['y']: batch_y,
    #         })
    #
    #         # Validation
    #         if i % every_n_iter == 0:
    #             _logging_hook(sess, saver, i, result, models)
    #
    #         if result['loss'] < EARLY_STOP_TRAIN_LOSS:
    #             break
    # except:
    #     traceback.print_exc()

    _logging_hook(sess, saver, i, result, models, final=True)
