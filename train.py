import argparse
import os.path
import sys
import time
import model
import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = None
input_dim = 32*32*3 # Dimension of the input
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

def read_and_decode(filename_queue):
    '''
    To read the tfrecords file
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([input_dim])

    #image = tf.reshape(image, shape=[32, 32, 3])
    image = tf.cast(image, tf.float32) * (1. / 255)

    label = tf.cast(features['label'], tf.int32)

    return image, label

def inputs(batch_size, num_epochs, filename):
    '''
    To create batches of training data to be fed to the net during training
    '''
    if not num_epochs:
        num_epochs = None
    print(filename)
    #filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue)

        images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2, capacity=10000 + 3 * batch_size, min_after_dequeue=1000)

    return images, sparse_labels

def accuracy(prediction, correct_answer):
    '''
    Calculates the accuracy during training and validation
    '''
    prediction = tf.argmax(prediction, dimension=1)
    equality = tf.equal(prediction, correct_answer)
    return tf.reduce_mean(tf.cast(equality, tf.float32))

def run_training():
    '''
    The function which builds the graph and conducts training and validation
    '''
    with tf.Graph().as_default():
        images, labels = inputs(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, filename=os.path.join(FLAGS.train_dir, TRAIN_FILE))
        logits = model.inference(images, reuse=False, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.dim)

        labels = tf.to_int64(labels)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        loss = tf.add_n([ce_loss], regularization_loss)
        pred = tf.nn.softmax(logits)
        train_acc = accuracy(pred, labels)

        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

        # Validation
        vali_images, vali_labels = inputs(batch_size=FLAGS.batch_size, num_epochs=4*FLAGS.num_epochs, filename=os.path.join(FLAGS.train_dir, VALIDATION_FILE))
        vali_logits = model.inference(vali_images, reuse=True, bn_epsilon=FLAGS.bn_epsilon, dim=FLAGS.dim)
        vali_labels = tf.to_int64(vali_labels)
        vali_pred = tf.nn.softmax(vali_logits)
        vali_accuracy = accuracy(vali_pred, vali_labels)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()

        saver = tf.train.Saver()
        if not os.path.exists(FLAGS.save_dir):
            os.makedirs(FLAGS.save_dir)

        save_path = os.path.join(FLAGS.save_dir, FLAGS.weights_file)
        try:
            print('Trying to restore checkpoint')
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.save_dir)
            saver.restore(sess, save_path=last_chk_path)
            print('Restored checkpoint from:', last_chk_path)
        except:
            print('No checkpoint found. Initializing variables..')
            sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tr_acc_list = []
        val_acc_list = []
        iters_list = []
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value, acc = sess.run([train_op, loss, train_acc])

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    vali_acc = sess.run([vali_accuracy])
                    print('Step %d: loss = %.2f Train Accuracy = %.3f Vali Accuracy = %.3f (%.3f sec)' % (step, loss_value, acc, vali_acc[0], duration))
                    tr_acc_list.append(acc)
                    val_acc_list.append(vali_acc)
                    iters_list.append(step)
                step += 1

                if step % FLAGS.max_steps == 0:
                    saver.save(sess, save_path=save_path, global_step=step)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(iters_list, tr_acc_list)
        plt.plot(iters_list, val_acc_list)
        plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
        plt.savefig('accuracy.png')



def main(_):
  run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs of training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of units in the penultimate hidden layer.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--max_steps', type=int, default=30000, help='Max no of steps before terminating training')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of filters')
    parser.add_argument('--train_dir', type=str, default='cifar10/', help='Directory with the training data.')
    parser.add_argument('--save_dir', type=str, default='saved_model/', help='Directory where models are saved.')
    parser.add_argument('--bn_epsilon', type=float, default=0.001, help='Epsilon for batch normalization.')
    parser.add_argument('--weights_file', type=str, default='cifar10_weights', help='Name of the saved weights file.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)