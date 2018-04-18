import os

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_epochs = 100
output_layer_size = 10
hidden_layer_size = 100
hidden_layer_number = 6
learning_rate = 0.01
batch_size = 100
steps = num_epochs * batch_size


def custom_model_fn(features, labels, mode):
    """Model function for PA1"""
    # Write your custom layer
    # Input Layer
    # You also can use 1 x 784 vector
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    input_layer = features["x"]

    hidden_layers = [
        tf.layers.dense(input_layer, hidden_layer_size, activation=tf.nn.relu)
    ]
    for i in range(1, hidden_layer_number):
        hidden_layers.append(
            tf.layers.dense(
                hidden_layers[i - 1], hidden_layer_size,
                activation=tf.nn.relu))

    # Output logits Layer
    logits = tf.layers.dense(hidden_layers[-1], output_layer_size)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses."custom loss function" # Refer to tf.losses
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train."custom optimizer" # Refer to tf.train

        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate * 0.01)
        eval_metric_ops = {"accuracy": accuracy}
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
        tensors_to_log = {"loss": loss, "accuracy": acc}
        # tensors_to_log = {}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[logging_hook])
    else:
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('PA1-data/train.npy')
    dataset_eval = np.load('PA1-data/valid.npy')
    test_data = np.load('PA1-data/test.npy')

    train_data = dataset_train[:, :784]
    # import random
    # f = random.randint(1, 10000)
    # for i in range(28):
    #     row = train_data[f, i * 28:(i + 1) * 28]
    #     print(''.join('■' if c > 0.67 else ('□' if c > 0.33 else ' ')
    #                   for c in row) + '|')
    # print(dataset_train[f, 784])
    #
    # from PIL import Image
    #
    # img = Image.new('RGB', (28, 28))
    # img.putdata([(int(i * 255), int(i * 255), int(i * 255)) for i in train_data[f, :]])
    # img.save('test.png')

    # assert False
    train_labels = dataset_train[:, 784].astype(np.int32)
    eval_data = dataset_eval[:, :784]
    eval_labels = dataset_eval[:, 784].astype(np.int32)

    # Save model and checkpoint
    classifier = tf.estimator.Estimator(
        model_fn=custom_model_fn, model_dir="./PA1-model-old")

    # Train the model. You can train your model with specific batch size and epoches
    for i in range(100):
        train_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=1,
            shuffle=True)
        classifier.train(input_fn=train_input, steps=steps)
        train_results = classifier.evaluate(input_fn=train_input)

        # Eval the model. You can evaluate your trained model with validation data
        eval_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input)
        print('>>')
        print(train_results)
        print(eval_results)

    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    # pred_input = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": test_data}, shuffle=False)
    # pred_results = classifier.predict(input_fn=pred_input)
    # # print(list(pred_results))
    # print(type(list(pred_results)))
    # result = np.asarray([x.values()[1] for x in list(pred_results)])
    ## ----------------------------------------- ##

    # np.save('20110727.npy', result)
    print('Done!')
