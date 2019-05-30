import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

'''
filter_weight = tf.get_variable('weight',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))

bias = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
bias = tf.nn.bias_add(conv,bias)

actived_conv = tf.nn.relu(bias)
pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
'''
IMPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

BATCH_SIZE = 100
REGULARIZER_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000


def inference(input_tensor, train, regularizer):
    # 第一层卷积层:28*28*32  过滤器边长5，步长1，填充0 out:28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    # 第二层池化，池化尺寸：28*28*32，填充0，步长2，out:14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层卷积：14*14*32，边长5，填充0，输出14*14*64
    with tf.variable_scope('layer2-conv2'):
        weight = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
    # 第四层池化：14*14*64，out:7*7*64
    with tf.name_scope('layer2-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 获取全连接层的输入
    #其中pool_shappe = [100,7,7,64],100 为batch_size
    pool_shape = pool2.get_shape().as_list()
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [pool_shape[0], node])

    with tf.variable_scope('layer5-fc'):
        fc1_weight = tf.get_variable('weight', [node, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_bias = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weight))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc'):
        fc2_weight = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weight))

        logit = tf.matmul(fc1, fc2_weight) + fc2_bias
    return logit


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x-input')

    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = inference(x, train=True, regularizer=regularizer)

    # 定义滑动平均
    global_step = tf.Variable(0, trainable=False)
    variable = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variabable_op = variable.apply(tf.trainable_variables())

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 指数衰减learning_rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    with tf.control_dependencies([train_step, variabable_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshapes_xs = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            _, loss_va, step = sess.run([train_op, loss, global_step], feed_dict={x: reshapes_xs, y_: ys})

            if i % 1000 == 0:
                print('after %d training,loss is %g' % (step, loss_va))
                saver.save(sess, './model/models.ckpt', global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(r'C:\\Users\Administrator\Desktop\PythonPRO\0528\DATA', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
