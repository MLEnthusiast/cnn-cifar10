import cPickle
import os
import tarfile
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
data_dir = 'cifar10/'


def download_and_extract(data_dir):
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME), 'r:gz').extractall(data_dir)

def _get_file_names():
    '''
    Reads the filenames
    '''
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
    file_names['validation'] = ['validation_batch']
    file_names['eval'] = ['test_batch']
    return file_names

def read_pickle_from_file(filename):
    '''
    Unpickles the downloaded files
    '''
    with tf.gfile.Open(filename, 'r') as f:
        data_dict = cPickle.load(f)
    return data_dict

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(input_files, output_file):
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            images = data_dict['data']
            labels = data_dict['labels']
            num_entries_in_batch = len(labels)
            print(num_entries_in_batch)
            for idx in range(num_entries_in_batch):
                image_raw = images[idx].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(int(labels[idx])),
                    'image_raw': _bytes_feature(image_raw)
                }))
                record_writer.write(example.SerializeToString())

def main():
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        convert_to_tfrecord(input_files, output_file)
    print('Done!')

if __name__ == '__main__':
    main()