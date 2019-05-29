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
#模型复杂度的正则化在损失函数中的系数
REGULARIZATION_RATE = 0.0001
#训练次数
TRAINING_STEPS = 30000

#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


#一个辅助函数帮助处理神经网络的所有输入，以及前向传播过程,在这里现在设定的是三层神经网络
#avg_class 滑动平均类
def inference(input_tensor, avg_class,weight,biasses1,weight2,biasses2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight)+biasses1)
        return tf.matmul(layer1,weight2)+biasses2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight))+biasses1)
        return tf.matmul(layer1,avg_class.average(weight2))+biasses2

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-output')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev=0.1))
    biasses1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev=0.1))
    biasses2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y = inference(x,None,weight1,biasses1,weight2,biasses2)

    #定义存储训练论输的变量，不可训练，该变量不需要进行滑动平滑计算
    global_steps = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_steps)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #正向传递结果
    average_y = inference(x,variable_averages,weight1,biasses1,weight2,biasses2)

    #计算交叉熵，求平均
    corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_mean = tf.reduce_mean(corss_entropy)
    #l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算权重的正则化损失，不使用偏置
    regularization = regularizer(weight1)+regularizer(weight2)
    #模型总损失
    loss = cross_mean + regularization
    #设置衰减速率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_steps,mnist.train.num_examples,LEARNING_RATE_DECAY)
    #优化算法，优化步骤，global——steps，优化内容：loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_steps)

    #对于每次训练，regularization每次都会发生变化，那么未来简化计算操作，TensorFlow提供了tf.control_dependencies and  tf.group()
    #两种机制，所以现在
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op =tf.no_op(name='train')

    correct_prediction =tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuarcy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed ={
            x:mnist.validation.images,
            y_:mnist.validation.labels
        }
        test_feed = {
            x:mnist.test.images,
            y_:mnist.test.labels
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuarcy,feed_dict=validate_feed)
                print('after %d training steps,validation accuarcy,using average model is %g' %(i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

    test_Acc = sess.run(accuarcy,feed_dict=test_feed)
    print('after %d training steps,validation accuarcy,using average model is %g' %(TRAINING_STEPS,test_Acc))

def main(argv=None):
    mnist = input_data.read_data_sets(r'C:\\Users\Administrator\Desktop\PythonPRO\0528\DATA',one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()

