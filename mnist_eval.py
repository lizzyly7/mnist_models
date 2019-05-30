import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import MNIST_p

EVAL_INTERVAL_SEC = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, MNIST_p.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, MNIST_p.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        y = MNIST_p.second_update_inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(MNIST_p.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state('./')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('after %s training steps,validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('no checkpoint found')
                    return
            time.sleep(EVAL_INTERVAL_SEC)


def main(argv=None):
    mnist = input_data.read_data_sets(r'C:\\Users\Administrator\Desktop\PythonPRO\0528\DATA', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
