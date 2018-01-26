import train
import tensorflow as tf
import numpy as np
import argparse
import os.path
import sys
import time
import model

TEST_FILE = 'eval.tfrecords'

def run_test():
    with tf.Graph().as_default():
        # Testing
        test_images, test_labels = train.inputs(batch_size=FLAGS.batch_size, num_epochs=1, filename=os.path.join(FLAGS.train_dir, TEST_FILE))
        test_logits = model.inference(test_images, reuse=False, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.dim)
        test_labels = tf.to_int64(test_labels)
        test_pred = tf.nn.softmax(test_logits)
        test_accuracy = train.accuracy(test_pred, test_labels)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(init_op)
        saver.restore(sess, os.path.join(FLAGS.save_dir, 'cifar_10_weights-10000'))
        print('Model restored!')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        test_acc_list = []
        try:
            while not coord.should_stop():
                test_acc = sess.run([test_accuracy])
                test_acc_list.append(test_acc[0])
        except tf.errors.OutOfRangeError:
            f_acc = np.mean(test_acc_list)
            print('Final accuracy %.3f' % f_acc)
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

def main(_):
    run_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--train_dir', type=str, default='cifar10/', help='Directory with the test data.')
    parser.add_argument('--bn_epsilon', type=float, default=0.001, help='Epsilon for batch normalization.')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of filters')
    parser.add_argument('--save_dir', type=str, default='saved_model/', help='Directory where models are saved.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)