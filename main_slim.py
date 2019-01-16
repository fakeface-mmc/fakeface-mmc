import os
import time
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model_slim import model 
from datetime import datetime
from tensorflow.contrib import layers
from random import shuffle
from glob import glob
from config import get_args
from logger import Logger


def data_load(data_dir):
    trn_r, trn_g = [], []
    val_r, val_g = [], []
    tst_r, tst_g = [], []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                filepath = os.path.join(root, file)
                if "00_" in root:
                    if "train" in root:
                        trn_r.append(filepath)
                    elif "val" in root:
                        val_r.append(filepath)
                    elif "test" in root:
                        tst_r.append(filepath)
                elif "01_" in root:
                    if "train" in root:
                        trn_g.append(filepath)
                    elif "val" in root:
                        val_g.append(filepath)
                    elif "test" in root:
                        tst_g.append(filepath)

    return trn_r, val_r, tst_r, trn_g, val_g, tst_g 


if __name__ == '__main__':
    # set environment
    FLAGS = get_args()
    NOW = FLAGS.load_time if FLAGS.load_time else datetime.now().strftime(r'%y-%m-%d_%H-%M')

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.log_dir)
    tl.files.exists_or_mkdir(FLAGS.graph_dir)

    logger = Logger(log_path=os.path.join(FLAGS.log_dir, NOW + '_log_train.txt'),
                    tb_path=os.path.join(FLAGS.graph_dir, datetime.now().strftime(r'%y-%m-%d_%H-%M')))
    logger.log_string(str(FLAGS)+'\n\n')
    logger.log_string(datetime.now().strftime('%y-%m-%d(%Hh%Mm)')+'\n')

    sess = tf.Session()

    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, FLAGS.img_resolution, FLAGS.img_resolution, 3], name='x') 
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # define inferences
    net_train = model(x, is_train=True, reuse=False)
    net_test = model(x, is_train=False, reuse=True)

    # define cost function and metric.
    y = net_train
    cost = tl.cost.cross_entropy(y, y_, name='xentropy') 
    cost = tf.reduce_mean(cost)

    # cost and accuracy for evaluation
    y2 = net_test
    cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
    correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, LayersConfig.tf_dtype))

    # define the optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 10000, 0.95, staircase=True)

    train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(cost, optimizer, variables_to_train=train_params)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None)

    # initialize all variables in the session
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    train_r_files, val_r_files, test_r_files, train_g_files, val_g_files, test_g_files = data_load(FLAGS.data_dir)
    logger.log_string('NUMBER OF IMAGES')
    logger.log_string(str(len(train_r_files)))
    logger.log_string(str(len(train_g_files))+'\n')

    num_batch_t = min(len(train_r_files), len(train_g_files)) // FLAGS.batch_size*2

    num_iter = 0
    min_err = np.inf

    if FLAGS.load_time is not None:
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.npz_to_load))
        num_iter = int(FLAGS.npz_to_load.split('_')[-2])

    for epoch in range(num_iter // num_batch_t, FLAGS.epoch):

        shuffle(train_r_files)
        shuffle(train_g_files)

        for t_idx in range(0, num_batch_t):
            start_time = time.time()

            # load training data
            train_r_images = tl.visualize.read_images(train_r_files[t_idx*FLAGS.batch_size//2:(t_idx+1)*FLAGS.batch_size//2], printable=False)
            train_r_images = tl.prepro.threading_data(train_r_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])

            train_g_images = tl.visualize.read_images(train_g_files[t_idx*FLAGS.batch_size//2:(t_idx+1)*FLAGS.batch_size//2], printable=False)
            train_g_images = tl.prepro.threading_data(train_g_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])

            train_r_labels = np.zeros(FLAGS.batch_size//2)
            train_g_labels = np.ones(FLAGS.batch_size//2)

            indices = np.arange(FLAGS.batch_size)

            np.random.shuffle(indices)
            X_train = np.concatenate([train_r_images, train_g_images])[indices]
            Y_train = np.concatenate([train_r_labels, train_g_labels])[indices]

            # load validation data
            batch_files = np.random.choice(val_r_files, FLAGS.batch_size//2)
            val_r_images = tl.visualize.read_images(batch_files, printable=False)
            val_r_images = tl.prepro.threading_data(val_r_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])

            batch_files = np.random.choice(val_g_files, FLAGS.batch_size//2)
            val_g_images = tl.visualize.read_images(batch_files, printable=False)
            val_g_images = tl.prepro.threading_data(val_g_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])     

            val_r_labels = np.zeros(FLAGS.batch_size//2)
            val_g_labels = np.ones(FLAGS.batch_size//2)

            indices = np.arange(FLAGS.batch_size)
            np.random.shuffle(indices)

            X_val = np.concatenate([val_r_images, val_g_images])[indices]
            Y_val = np.concatenate([val_r_labels, val_g_labels])[indices]

            err, ac, _ = sess.run([cost_test, acc, train_op], feed_dict={x: X_train, y_: Y_train})
            val_err, val_ac = sess.run([cost_test, acc], feed_dict={x: X_val, y_: Y_val})

            if num_iter % FLAGS.print_freq == 0 and num_iter > 0:
                logger.log_string("Epoch: [%2d/%2d] [%4d/%4d] (iter.: %5d) time: %4.4f, tr_loss: %.4f, tr_acc: %.2f, val_loss: %.4f, val_acc: %.2f"
                                  % (epoch, FLAGS.epoch, t_idx, num_batch_t, num_iter, time.time() - start_time, err, ac*100, val_err, val_ac*100)) 

            num_iter += 1

        # Validation at the end of every epoch
        tmp = FLAGS.batch_size
        if min(len(val_r_files), len(val_g_files)) < FLAGS.batch_size:
            FLAGS.batch_size = min(len(val_r_files), len(val_g_files))
        val_idxs = min(len(val_r_files), len(val_g_files)) // FLAGS.batch_size * 2
        val_loss, val_acc, num_batch_v = 0, 0, 0
        for v_idx in range(0, val_idxs):
            val_r_images = tl.visualize.read_images(val_r_files[v_idx*FLAGS.batch_size//2:(v_idx+1)*FLAGS.batch_size//2], printable=False)
            val_r_images = tl.prepro.threading_data(val_r_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])
            val_g_images = tl.visualize.read_images(val_g_files[v_idx*FLAGS.batch_size//2:(v_idx+1)*FLAGS.batch_size//2], printable=False)
            val_g_images = tl.prepro.threading_data(val_g_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])

            val_r_labels = np.zeros(FLAGS.batch_size//2) 
            val_g_labels = np.ones(FLAGS.batch_size//2) 
            X_val = np.concatenate([val_r_images, val_g_images])
            Y_val = np.concatenate([val_r_labels, val_g_labels])

            # X_val = X_val.astype(np.float32) / 255.
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val, y_: Y_val})

            val_loss += err
            val_acc += ac
            num_batch_v += 1

        FLAGS.batch_size = tmp

        val_loss, val_acc = val_loss / num_batch_v, val_acc / num_batch_v
        logger.log_string("[*] End of Epoch: [%2d/%2d] time: %4.4f, val_loss: %.4f, val_acc: %.2f" % (epoch, FLAGS.epoch, time.time() - start_time, val_loss, val_acc*100))

        # after "half" epochs, save model
        if val_loss < min_err and num_iter > FLAGS.epoch * num_batch_t // 2:
            min_err = val_loss
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, NOW + '_' + str(epoch) + '_' + str(num_iter) + '_minErr.ckpt'))
            logger.log_string("[*] Minimum validation error updated [%.4f]" % min_err)
            logger.log_string("[*] Saving [%d epoch] checkpoints SUCCESS!" % epoch)

        if epoch % FLAGS.save_interval == 0 and epoch is not 0:
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, NOW + '_' + str(epoch) + '_' + str(num_iter) + '_net.ckpt'))
            logger.log_string("[*] Saving [%d epoch] checkpoints SUCCESS!" % epoch)

    # Evaluation
    path = glob(os.path.join(FLAGS.checkpoint_dir, NOW + "*minErr.ckpt*"))
    if len(path) != 0:
        path, _ = os.path.splitext(path[-1])
        saver.restore(sess, path)
        logger.log_string("[*] Loading min-error model SUCCESS! [%s]" % os.path.basename(path))
    else:
        logger.log_string("[*] Cannot load model!")

    test_r_loss, test_g_loss = 0, 0
    test_r_acc, test_g_acc = 0, 0
    num_batch = min(len(test_r_files), len(test_g_files)) // FLAGS.batch_size
    for s_idx in range(num_batch):
        test_r_images = tl.visualize.read_images(test_r_files[s_idx*FLAGS.batch_size:(s_idx+1)*FLAGS.batch_size], printable=False)
        test_r_images = tl.prepro.threading_data(test_r_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])
        test_g_images = tl.visualize.read_images(test_g_files[s_idx*FLAGS.batch_size:(s_idx+1)*FLAGS.batch_size], printable=False)
        test_g_images = tl.prepro.threading_data(test_g_images, tl.prepro.imresize, size=[FLAGS.img_resolution, FLAGS.img_resolution])

        X_test = np.array(test_r_images)
        Y_test = np.zeros(FLAGS.batch_size)

        err, ac = sess.run([cost_test, acc], feed_dict={x: X_test, y_: Y_test})
        test_r_loss += err
        test_r_acc += ac

        X_test = np.array(test_g_images)
        Y_test = np.ones(FLAGS.batch_size)

        err, ac = sess.run([cost_test, acc], feed_dict={x: X_test, y_: Y_test})
        test_g_loss += err
        test_g_acc += ac

    if num_batch > 0:
        test_r_loss /= num_batch
        test_g_loss /= num_batch
        test_r_acc /= num_batch
        test_g_acc /= num_batch
        test_total_loss = (test_r_loss + test_g_loss) / 2
        test_total_acc = (test_r_acc + test_g_acc) / 2
        logger.log_string("[*] TEST loss: %.4f, acc: %.2f, r_loss: %.4f, r_acc: %.2f, g_loss: %.4f, g_acc: %.2f"
                          % (test_total_loss, test_total_acc*100, test_r_loss, test_r_acc*100, test_g_loss, test_g_acc*100))
