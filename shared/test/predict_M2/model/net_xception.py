from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import tensorflow as tf
slim = tf.contrib.slim

'''
==================================================================
Based on the Xception Paper (https://arxiv.org/pdf/1610.02357.pdf)
==================================================================

REGULARIZATION CONFIGURATION:
- weight_decay: 1e-5
- dropout: no dropout
- aux_loss: no aux loss

OPTIMIZATION CONFIGURATION (for Google JFT Dataset):
- optimizer: RMSProp
- momentum: 0.9
- initial_learning_rate: 0.001
- learning_rate_decay: 0.9 every 3/350 epochs (every 3M images; total 350M images per epoch)

'''

#input and labels

def model(img, label, phase):
    #===========ENTRY FLOW==============
    #Block 1
    net = slim.conv2d(img, 32, [3,3], stride=2, padding='valid', scope='block1_conv1') #x1을 입력 데이터로 바꾸면 됨.
    #net = slim.conv2d(x1, 32, [3,3], stride=1, padding='valid', scope='block1_conv1')
    net = slim.batch_norm(net, scope='block1_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block1_relu1')
    net = slim.conv2d(net, 64, [3,3], padding='valid', scope='block1_conv2')
    net = slim.batch_norm(net, scope='block1_bn2', is_training = phase)
    net = tf.nn.relu(net, name='block1_relu2')
    residual = slim.conv2d(net, 128, [1,1], stride=2, scope='block1_res_conv')
    residual = slim.batch_norm(residual, scope='block1_res_bn', is_training = phase)

    #Block 2
    net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv1')
    net = slim.batch_norm(net, scope='block2_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block2_relu1')
    net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv2')
    net = slim.batch_norm(net, scope='block2_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block2_max_pool')
    net = tf.add(net, residual, name='block2_add')
    residual = slim.conv2d(net, 256, [1,1], stride=2, scope='block2_res_conv')
    residual = slim.batch_norm(residual, scope='block2_res_bn', is_training = phase)

    #Block 3
    net = tf.nn.relu(net, name='block3_relu1')
    net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv1')
    net = slim.batch_norm(net, scope='block3_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block3_relu2')
    net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv2')
    net = slim.batch_norm(net, scope='block3_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block3_max_pool')
    net = tf.add(net, residual, name='block3_add')

    #===========MIDDLE FLOW===============
    for i in range(8):
        block_prefix = 'block%s_' % (str(i + 5))

        residual = net
        net = tf.nn.relu(net, name=block_prefix+'relu1')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv1')
        net = slim.batch_norm(net, scope=block_prefix+'bn1', is_training = phase)
        net = tf.nn.relu(net, name=block_prefix+'relu2')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv2')
        net = slim.batch_norm(net, scope=block_prefix+'bn2', is_training = phase)
        net = tf.nn.relu(net, name=block_prefix+'relu3')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv3')
        net = slim.batch_norm(net, scope=block_prefix+'bn3', is_training = phase)
        net = tf.add(net, residual, name=block_prefix+'add')


    #========EXIT FLOW============
    residual = slim.conv2d(net, 768, [1,1], stride=2, scope='block12_res_conv')
    residual = slim.batch_norm(residual, scope='block12_res_bn', is_training = phase)
    net = tf.nn.relu(net, name='block13_relu1')
    net = slim.separable_conv2d(net, 512, [3,3], scope='block13_dws_conv1')
    net = slim.batch_norm(net, scope='block13_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block13_relu2')
    net = slim.separable_conv2d(net, 768, [3,3], scope='block13_dws_conv2')
    net = slim.batch_norm(net, scope='block13_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block13_max_pool')
    net = tf.add(net, residual, name='block13_add')

    net = slim.separable_conv2d(net, 1024, [3,3], scope='block14_dws_conv1')
    net = slim.batch_norm(net, scope='block14_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block14_relu1')
    net = slim.separable_conv2d(net, 2048, [3,3], scope='block14_dws_conv2')
    net = slim.batch_norm(net, scope='block14_bn2', is_training = phase)
    net = tf.nn.relu(net, name='block14_relu2')
    print(net)

    #net = slim.avg_pool2d(net, [10,10], scope='block15_avg_pool')
    net = slim.avg_pool2d(net, [4,4], scope='block15_avg_pool') #입력 사이즈에 맞게 변환시켜주면 됨.
    #Replace FC layer with conv layer instead
    net = slim.conv2d(net, 2048, [1,1], scope='block15_conv1')
    logits = slim.conv2d(net, 2, [1,1], activation_fn=None, scope='block15_conv2')
    logits = tf.squeeze(logits, [1,2], name='block15_logits') #Squeeze height and width only
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))

    prediction = slim.softmax(logits, scope='Predictions')

    tf.summary.scalar('ce', cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    merge = tf.summary.merge_all()
    return  cross_entropy, accuracy, merge, prediction



def model_predict(img, phase):
    #===========ENTRY FLOW==============
    #Block 1
    net = slim.conv2d(img, 32, [3,3], stride=2, padding='valid', scope='block1_conv1') #x1을 입력 데이터로 바꾸면 됨.
    #net = slim.conv2d(x1, 32, [3,3], stride=1, padding='valid', scope='block1_conv1')
    net = slim.batch_norm(net, scope='block1_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block1_relu1')
    net = slim.conv2d(net, 64, [3,3], padding='valid', scope='block1_conv2')
    net = slim.batch_norm(net, scope='block1_bn2', is_training = phase)
    net = tf.nn.relu(net, name='block1_relu2')
    residual = slim.conv2d(net, 128, [1,1], stride=2, scope='block1_res_conv')
    residual = slim.batch_norm(residual, scope='block1_res_bn', is_training = phase)

    #Block 2
    net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv1')
    net = slim.batch_norm(net, scope='block2_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block2_relu1')
    net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv2')
    net = slim.batch_norm(net, scope='block2_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block2_max_pool')
    net = tf.add(net, residual, name='block2_add')
    residual = slim.conv2d(net, 256, [1,1], stride=2, scope='block2_res_conv')
    residual = slim.batch_norm(residual, scope='block2_res_bn', is_training = phase)

    #Block 3
    net = tf.nn.relu(net, name='block3_relu1')
    net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv1')
    net = slim.batch_norm(net, scope='block3_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block3_relu2')
    net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv2')
    net = slim.batch_norm(net, scope='block3_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block3_max_pool')
    net = tf.add(net, residual, name='block3_add')

    #===========MIDDLE FLOW===============
    for i in range(8):
        block_prefix = 'block%s_' % (str(i + 5))

        residual = net
        net = tf.nn.relu(net, name=block_prefix+'relu1')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv1')
        net = slim.batch_norm(net, scope=block_prefix+'bn1', is_training = phase)
        net = tf.nn.relu(net, name=block_prefix+'relu2')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv2')
        net = slim.batch_norm(net, scope=block_prefix+'bn2', is_training = phase)
        net = tf.nn.relu(net, name=block_prefix+'relu3')
        net = slim.separable_conv2d(net, 256, [3,3], scope=block_prefix+'dws_conv3')
        net = slim.batch_norm(net, scope=block_prefix+'bn3', is_training = phase)
        net = tf.add(net, residual, name=block_prefix+'add')


    #========EXIT FLOW============
    residual = slim.conv2d(net, 768, [1,1], stride=2, scope='block12_res_conv')
    residual = slim.batch_norm(residual, scope='block12_res_bn', is_training = phase)
    net = tf.nn.relu(net, name='block13_relu1')
    net = slim.separable_conv2d(net, 512, [3,3], scope='block13_dws_conv1')
    net = slim.batch_norm(net, scope='block13_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block13_relu2')
    net = slim.separable_conv2d(net, 768, [3,3], scope='block13_dws_conv2')
    net = slim.batch_norm(net, scope='block13_bn2', is_training = phase)
    net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block13_max_pool')
    net = tf.add(net, residual, name='block13_add')

    net = slim.separable_conv2d(net, 1024, [3,3], scope='block14_dws_conv1')
    net = slim.batch_norm(net, scope='block14_bn1', is_training = phase)
    net = tf.nn.relu(net, name='block14_relu1')
    net = slim.separable_conv2d(net, 2048, [3,3], scope='block14_dws_conv2')
    net = slim.batch_norm(net, scope='block14_bn2', is_training = phase)
    net = tf.nn.relu(net, name='block14_relu2')
    print(net)

    #net = slim.avg_pool2d(net, [10,10], scope='block15_avg_pool')
    net = slim.avg_pool2d(net, [4,4], scope='block15_avg_pool') #입력 사이즈에 맞게 변환시켜주면 됨.
    #Replace FC layer with conv layer instead
    net = slim.conv2d(net, 2048, [1,1], scope='block15_conv1')
    logits = slim.conv2d(net, 2, [1,1], activation_fn=None, scope='block15_conv2')
    logits = tf.squeeze(logits, [1,2], name='block15_logits') #Squeeze height and width only
    prediction = slim.softmax(logits, scope='Predictions')


    return prediction
