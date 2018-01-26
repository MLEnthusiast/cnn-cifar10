import model
import time
import argparse
import sys
import numpy as np
import tensorflow as tf
import cifar10
import matplotlib.pyplot as plt

FLAGS = None
IMG_SIZE = 32
CHANNELS = 3
max_iter = 20000
classes = 10

class Train(object):

    def __init__(self):
        self.placeholders()

    def placeholders(self):
        self.train_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, IMG_SIZE, IMG_SIZE, CHANNELS])
        self.train_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, classes])
        self.validation_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, IMG_SIZE, IMG_SIZE, CHANNELS])
        self.validation_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, classes])
        #self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def train_vali_graph(self):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

        train_logits = model.inference(inputs=self.train_image_placeholder, reuse=False, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.filter_dim)
        validation_logits = model.inference(inputs=self.validation_image_placeholder, reuse=True, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.filter_dim)

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.cross_entropy_loss(train_logits, self.train_label_placeholder)
        self.total_loss = tf.add_n([loss], regularization_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.total_loss)

        # Training accuracy
        train_pred = tf.nn.softmax(train_logits)
        self.train_accuracy = self.get_accuracy(train_pred, self.train_label_placeholder)

        # Validation accuracy
        validation_pred = tf.nn.softmax(validation_logits)
        self.validation_accuracy = self.get_accuracy(validation_pred, self.validation_label_placeholder)

        # Test
        test_logits = model.inference(inputs=self.validation_image_placeholder, reuse=True, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.filter_dim)
        test_pred = tf.nn.softmax(test_logits)
        self.test_accuracy = self.get_accuracy(test_pred, self.validation_label_placeholder)

    def train(self):
        self.train_vali_graph()
        images_train, cls_train, labels_train = cifar10.load_training_data()
        images_validation, cls_validation, labels_validation = cifar10.load_validation_data()
        images_test, cls_test, labels_test = cifar10.load_test_data()

        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init_op)
            tr_acc_list = []
            val_acc_list = []
            iters_list = []
            for i in range(max_iter):
                x_batch_train, y_batch_labels = self.random_batch(images_train, labels_train)
                _, cost, tr_acc = sess.run([self.train_op, self.total_loss, self.train_accuracy], feed_dict={
                    self.train_image_placeholder: x_batch_train,
                    self.train_label_placeholder: y_batch_labels
                })

                if i % 1000 == 0:
                    x_batch_vali, y_batch_vali = self.random_batch(images_validation, labels_validation)
                    vali_acc = sess.run([self.validation_accuracy], feed_dict={
                        self.validation_image_placeholder: x_batch_vali,
                        self.validation_label_placeholder: y_batch_vali
                    })

                    print('Step %d Loss=%.3f Training Accuracy = %.3f Validation Accuracy = %.3f' % (i, cost, tr_acc, vali_acc[0]))
                    tr_acc_list.append(tr_acc)
                    val_acc_list.append(vali_acc)
                    iters_list.append(i)
            print('Optimization Done!')

            start = 0
            num_test = len(images_test)
            final_acc = []
            while start < num_test:
                end = min(start + FLAGS.batch_size, num_test)
                test_acc = sess.run([self.test_accuracy], feed_dict={
                    self.validation_image_placeholder: images_test[start:end, :, :, :],
                    self.validation_label_placeholder: labels_test[start:end, :]
                })
                final_acc.append(test_acc)
                start = end
            f_acc = np.mean(final_acc)
            print('Final test accuracy = %.3f' % f_acc)
            fig = plt.figure(figsize=(10, 10))
            plt.plot(iters_list, tr_acc_list)
            plt.plot(iters_list, val_acc_list)
            plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
            plt.savefig('accuracy.png')

    def random_batch(self, images, labels):
        num_images = len(images)

        idx = np.random.choice(num_images, size=FLAGS.batch_size, replace=False)
        x_batch = images[idx, :, :, :]
        y_batch = labels[idx, :]

        return x_batch, y_batch


    def cross_entropy_loss(self, logits, labels):
        labels = tf.cast(labels, dtype=tf.int32)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cross_entropy_loss

    def get_accuracy(self, predictions, labels):
        predictions = tf.argmax(predictions, 1)
        correct_answer = tf.argmax(labels, 1)
        equality = tf.equal(predictions, correct_answer)
        accuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))
        return accuracy

def main(_):
    train = Train()
    train.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--bn_epsilon', type=float, default=0.001, help='Epsilon for batch normalization')
    parser.add_argument('--filter_dim', type=int, default=32, help='Dimension of filters')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)