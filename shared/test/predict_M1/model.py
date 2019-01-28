from tensorlayer.layers import *
import tensorflow as tf

# model for discriminating gan images from others
def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        n = InputLayer(x, name='input')

        # cnn
        n = Conv2d(n, 32, (5, 5), (2, 2), act=tf.nn.relu, padding='SAME', name='cnn1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool2')
        n = Conv2d(n, 64, (5, 5), (2, 2),  act=tf.nn.relu, padding='SAME', name='cnn2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn2')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool2')
        n = Conv2d(n, 128, (5, 5), (2, 2),  act=tf.nn.relu, padding='SAME', name='cnn3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn3')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool2')
        # mlp
        n = FlattenLayer(n, name='flatten')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop1')
        n = DenseLayer(n, 512, act=tf.nn.relu, name='relu1')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop2')
        n = DenseLayer(n, 256, act=tf.nn.relu, name='relu2')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop3')
        n = DenseLayer(n, 2, act=tf.identity, name='output')
    return n