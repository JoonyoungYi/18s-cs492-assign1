import tensorflow as tf

# Hyperparameters
input_layer_size = 784
hidden_layer_size = 50
output_layer_size = 10

# Download MNIST dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Build the model
x = tf.placeholder(tf.float32, [None, input_layer_size], 'x')
y = tf.placeholder(tf.int32, [None], 'y')

hidden1 = tf.layers.dense(x, hidden_layer_size, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, hidden_layer_size, activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, hidden_layer_size, activation=tf.nn.relu)
hidden4 = tf.layers.dense(hidden3, hidden_layer_size, activation=tf.nn.relu)
hidden5 = tf.layers.dense(hidden4, hidden_layer_size, activation=tf.nn.relu)
hidden6 = tf.layers.dense(hidden5, hidden_layer_size, activation=tf.nn.relu)
hidden7 = tf.layers.dense(hidden6, hidden_layer_size, activation=tf.nn.relu)
output = tf.layers.dense(hidden7, output_layer_size)

pred = tf.argmax(output, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

# Checkpoint
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, './logs/ckpt/model-10000.ckpt')

feed_dict = {x: mnist.test.images, y: mnist.test.labels}
test_accuracy = sess.run(acc, feed_dict)
print('\n\nTest accuracy:', test_accuracy)
