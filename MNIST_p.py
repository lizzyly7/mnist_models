from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

'''
mnist = input_data.read_data_sets(r'C:\\Users\Administrator\Desktop\PythonPRO\0528\DATA',one_hot=True)
#check the size of train dataset
print('training data size:',mnist.train.num_examples)
print('validate dataset size',mnist.validation.num_examples)
print('test data size', mnist.test.num_examples)

print('example training data:',mnist.train.images[0])
print('example train data',mnist.train.labels[0])

batch_size = 100
#select input data from trian dataset while a batch each time
xs,ys = mnist.train.next_batch(batch_size)

#in mnist dataset, each picture is described as 28*28 pixels in total 784, the number is white(0) to black(1) probability
print('X shape:', xs.shape)
print('Y shape:', ys.shape)
'''
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
# 模型复杂度的正则化在损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练次数
TRAINING_STEPS = 30000

# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


# 一个辅助函数帮助处理神经网络的所有输入，以及前向传播过程,在这里现在设定的是三层神经网络
# avg_class 滑动平均类
def inference(input_tensor, avg_class, weight, biasses1, weight2, biasses2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + biasses1)
        return tf.matmul(layer1, weight2) + biasses2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight)) + biasses1)
        return tf.matmul(layer1, avg_class.average(weight2)) + biasses2


def update_inference(input_tensor, regularizer):
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weight', [INPUT_NODE, LAYER_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)
    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [LAYER_NODE, OUTPUT_NODE],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        biass = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weight) + biass

    return layer2, weights, weight


def get_weight_vatiable(shape, regularizer):
    weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight


def second_update_inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weight = get_weight_vatiable([INPUT_NODE, LAYER_NODE], regularizer)
        bias = tf.get_variable('bias', [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + bias)
    with tf.variable_scope('layer2'):
        weight = get_weight_vatiable([LAYER_NODE, OUTPUT_NODE], regularizer)
        bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weight) + bias
    return layer2


def train(mnist):
    '''

     x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    biasses1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biasses2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weight1, biasses1, weight2, biasses2)
    '''
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = second_update_inference(x, regularizer)

    # 定义存储训练轮数的变量，不可训练，该变量不需要进行滑动平滑计算
    global_steps = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 正向传递结果
    # average_y = inference(x, variable_averages, weight1, biasses1, weight2, biasses2)

    # 计算交叉熵，求平均
    corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_mean = tf.reduce_mean(corss_entropy)

    # 计算权重的正则化损失，不使用偏置(updated_inference)
    # regularization = regularizer(weight1) + regularizer(weight2)

    # 模型总损失
    loss = cross_mean + tf.add_n(tf.get_collection('losses'))
    # 设置衰减速率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 优化算法，优化步骤，global——steps，优化内容：loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_steps)

    # 对于每次训练，regularization每次都会发生变化，那么未来简化计算操作，TensorFlow提供了tf.control_dependencies and  tf.group()
    # 两种机制，所以现在
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    #accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_steps], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print('after %d training steps.loss on training, batch is %g.' % (step, loss_value))
                saver.save(sess, './model.ckpt',global_step=global_steps)
        '''
        #without batch training
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuarcy, feed_dict=validate_feed)
                print('after %d training steps,validation accuarcy,using average model is %g' % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        saver.save(sess, '/model.ckpt')
        # 对于测试数据集的图片进行测试
        test_Acc = sess.run(accuarcy, feed_dict=test_feed)
        print('after %d training steps,test accuarcy,using average model is %g' % (TRAINING_STEPS, test_Acc))
        '''


def main(argv=None):
    mnist = input_data.read_data_sets(r'C:\\Users\Administrator\Desktop\PythonPRO\0528\DATA', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
