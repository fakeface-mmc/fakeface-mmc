from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range
import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse, values=[x]), \
        slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], activation_fn=None), \
        slim.arg_scope([slim.batch_norm], decay=0.9, scale=True, epsilon=1e-3):

        #===========ENTRY FLOW==============
        #Block 1
        net = slim.conv2d(x, 32, [3, 3], stride=2, padding='valid', scope='block1_conv1')
        net = slim.batch_norm(net, scope='block1_bn1', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.conv2d(net, 64, [3, 3], padding='valid', scope='block1_conv2')
        net = slim.batch_norm(net, scope='block1_bn2', activation_fn=tf.nn.relu, is_training=is_train)
        residual = slim.conv2d(net, 128, [1, 1], stride=2, scope='block1_res_conv')
        residual = slim.batch_norm(residual, scope='block1_res_bn', is_training=is_train)

        #Block 2
        net = slim.separable_conv2d(net, 128, [3, 3], depth_multiplier=1, scope='block2_dws_conv1')
        net = slim.batch_norm(net, scope='block2_bn1', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.separable_conv2d(net, 128, [3, 3], depth_multiplier=1, scope='block2_dws_conv2')
        net = slim.batch_norm(net, scope='block2_bn2', is_training=is_train)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block2_max_pool')
        net = tf.add(net, residual, name='block2_add')
        residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
        residual = slim.batch_norm(residual, scope='block2_res_bn', is_training=is_train)

        #Block 3
        net = tf.nn.relu(net, name='block3_relu1')
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope='block3_dws_conv1')
        net = slim.batch_norm(net, scope='block3_bn1', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope='block3_dws_conv2')
        net = slim.batch_norm(net, scope='block3_bn2', is_training=is_train)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
        net = tf.add(net, residual, name='block3_add')

        #===========MIDDLE FLOW===============
        for i in range(8):
            block_prefix = 'block%s_' % (str(i + 5))

            residual = net
            net = tf.nn.relu(net, name=block_prefix+'relu1')
            net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope=block_prefix+'dws_conv1')
            net = slim.batch_norm(net, scope=block_prefix+'bn1', activation_fn=tf.nn.relu, is_training=is_train)
            net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope=block_prefix+'dws_conv2')
            net = slim.batch_norm(net, scope=block_prefix+'bn2', activation_fn=tf.nn.relu, is_training=is_train)
            net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope=block_prefix+'dws_conv3')
            net = slim.batch_norm(net, scope=block_prefix+'bn3', is_training=is_train)
            net = tf.add(net, residual, name=block_prefix+'add')

        #========EXIT FLOW============
        residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block12_res_conv')
        residual = slim.batch_norm(residual, scope='block12_res_bn', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope='block13_dws_conv1')
        net = slim.batch_norm(net, scope='block13_bn1', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope='block13_dws_conv2')
        net = slim.batch_norm(net, scope='block13_bn2', is_training=is_train)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block13_max_pool')
        net = tf.add(net, residual, name='block13_add')
        net = tf.nn.relu(net)

        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1, scope='block14_dws_conv1')
        net = slim.batch_norm(net, scope='block14_bn1', activation_fn=tf.nn.relu, is_training=is_train)
        net = slim.separable_conv2d(net, 512, [3, 3], depth_multiplier=1, scope='block14_dws_conv2')
        net = slim.batch_norm(net, scope='block14_bn2', activation_fn=tf.nn.relu, is_training=is_train)

        net = tf.reduce_mean(net, axis=[1, 2], name='block15_avg_pool')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 2, activation_fn=None, scope='fc')
        return net
