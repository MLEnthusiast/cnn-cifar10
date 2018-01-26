import tensorflow as tf
import tensorflow.contrib.slim as slim

KEEP_PROB = 0.5
def create_weights(name, filter_shape, initializer):
    return tf.get_variable(name=name+'.weights', shape=filter_shape, initializer=initializer)

def create_biases(name, filter_shape, initializer=tf.constant_initializer(0)):
    return tf.get_variable(name=name+'.biases', shape=filter_shape, initializer=initializer)

def batch_normalization(name, inputs, dims, bn_epsilon):
    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    beta = tf.get_variable(name+'.beta', dims, tf.float32, initializer=tf.constant_initializer(0., tf.float32))
    gamma = tf.get_variable(name+'.gamma', dims, tf.float32, initializer=tf.constant_initializer(1., tf.float32))

    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, bn_epsilon)

def conv_bn_relu_layer(inputs, name, filter_size, input_dim, output_dim, stride, bn_epsilon, bn=True):
    weights = create_weights(name=name, filter_shape=[filter_size, filter_size, input_dim, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
    biases = create_biases(name=name, filter_shape=[output_dim])

    conv = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.bias_add(conv, biases)
    if bn:
        output = batch_normalization(name=name, inputs=output, dims=output_dim, bn_epsilon=bn_epsilon)
    output = tf.nn.relu(output)

    return output

def fully_connected_layer(inputs, name, input_dim, output_dim, activation_fn=None):
    weights = create_weights(name=name, filter_shape=[input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = create_biases(name=name, filter_shape=[output_dim])

    output = tf.matmul(inputs, weights)
    output = tf.nn.bias_add(output, biases)
    if activation_fn is not None:
        output = tf.nn.relu(output)
    return output
'''
def inference(inputs, reuse, bn_epsilon, dim, hidden_dim):
    with tf.variable_scope('inference_graph', reuse=reuse):
        output = tf.reshape(inputs, shape=[-1, 32, 32, 3])
        output = tf.image.resize_image_with_crop_or_pad(output, target_height=24, target_width=24)
        output = conv_bn_relu_layer(inputs=output, name='conv1', filter_size=3, input_dim=3, output_dim=dim, stride=2, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv2', filter_size=3, input_dim=dim, output_dim=2*dim, stride=2, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv3', filter_size=3, input_dim=2*dim, output_dim=4*dim, stride=2, bn_epsilon=bn_epsilon)
        output = tf.reshape(output, shape=[-1, 3*3*4*dim])
        output = fully_connected_layer(inputs=output, name='fc1', input_dim=3*3*4*dim, output_dim=hidden_dim)
        output = tf.nn.dropout(output, keep_prob=1 if reuse else 0.5)
        output = fully_connected_layer(inputs=output, name='out', input_dim=hidden_dim, output_dim=10)
    return output
'''
def inference(inputs, reuse, bn_epsilon, dim):
    with tf.variable_scope('inference_graph', reuse=reuse):
        output = tf.reshape(inputs, shape=[-1, 32, 32, 3])
        output = tf.image.resize_image_with_crop_or_pad(output, target_height=24, target_width=24)
        output = conv_bn_relu_layer(inputs=output, name='conv1', filter_size=3, input_dim=3, output_dim=dim, stride=2, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv2', filter_size=3, input_dim=dim, output_dim=2*dim, stride=2, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv3', filter_size=3, input_dim=2*dim, output_dim=4*dim, stride=2, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv4', filter_size=3, input_dim=4*dim, output_dim=4*dim, stride=1, bn_epsilon=bn_epsilon)
        output = conv_bn_relu_layer(inputs=output, name='conv5', filter_size=3, input_dim=4*dim, output_dim=10, stride=1, bn_epsilon=bn_epsilon)
        output = tf.reduce_mean(output, [1, 2])  # global average pooling
    return output
