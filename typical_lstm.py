import tensorflow as tf

import numpy as np

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYER = 2

TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(x, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYER)])
    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = output[:, -1, :]

    prediction = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return prediction, None, None
    loss = tf.losses.mean_squared_error(labels=y,predictions=prediction)

    train_op = tf.contrib.layers.optimize_loss(
        loss,tf.train.get_global_step(),optimizer='Adagrad',learning_rate=0.1
    )
    return prediction,loss, train_op

def train(sess, train_x, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model'):
        prediction, loss, train_op = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, loss_va = sess.run([train_op, loss])
        if i % 100 == 0:
            print('after %d steps,loss is %g' % (i, loss_va))


def eva(sess, train_x, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model', reuse=True):
        predictions, _, _ = lstm_model(train_x, [0.0], False)

    prediction = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([predictions, y])
        prediction.append(p)
        labels.append(l)

    prediction = np.array(prediction).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((prediction - labels) ** 2).mean(axis=0))
    #print(rmse)
    #print('mean sqrt error is %f' % rmse)

    plt.figure()
    plt.plot(prediction, label='prediction')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_x, test_y = generate_data(
    np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
print(np.shape(train_x),np.shape(train_y))
with tf.Session() as sess:
    train(sess, train_x, train_y)
    eva(sess, test_x, test_y)
