import argparse
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import time
import os
from model import *
from datetime import datetime
from tensorflow.contrib import layers
from random import shuffle
from glob import glob
import random

# Arguments for training
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20, help="Epoch to train [20]")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate of adam [0.00001]")
parser.add_argument("--batch_size", type=int, default=50, help="The number of batch images [50]")
parser.add_argument("--data_dir", default='D:/data/rnd_gan', help="The data directory")
parser.add_argument("--log_dir", default="log", help="The directory for logging")
parser.add_argument("--checkpoint_dir", default="npz", help="Directory name to save the checkpoints [npz]")
parser.add_argument("--img_resolution", type=int, default=64, help="The resolution of images [64]")
parser.add_argument("--print_freq", type=int, default=1, help="The interval of batches to print [1]")
parser.add_argument("--save_interval", type=int, default=10, help="The interval of saving npz [10]")
parser.add_argument('--mode', default='train', help='mode [train, test]')

# Arguments for loading 
parser.add_argument("--load_time", default=None, help="load time")
parser.add_argument("--npz_to_load", default='None', help="used when load the pretrained model")
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_EPOCH = FLAGS.epoch
LEARNING_RATE = FLAGS.learning_rate
DATA_DIR = FLAGS.data_dir
RESOLUTION = FLAGS.img_resolution
CHECKPOINT_DIR = FLAGS.checkpoint_dir
LOG_DIR = FLAGS.log_dir
PRINT_FREQ = FLAGS.print_freq
LOAD_TIME = FLAGS.load_time
MODE = FLAGS.mode
NPZ_TO_LOAD = FLAGS.npz_to_load
SAVE_INTERVAL = FLAGS.save_interval

if LOAD_TIME is not None: now = LOAD_TIME
else: now = datetime.now().strftime('%y-%m-%d(%Hh%Mm)')

LOG_FOUT = open(os.path.join(LOG_DIR, now + '_log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
LOG_FOUT.write(datetime.now().strftime('%y-%m-%d(%Hh%Mm)')+'\n')

tl.files.exists_or_mkdir(CHECKPOINT_DIR)
tl.files.exists_or_mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, now + '_log_train.txt'), 'a')
LOG_FOUT.write('****' + datetime.now().strftime('%y-%m-%d(%Hh%Mm)') + '****' + '\n')
LOG_FOUT.write(str(FLAGS)+'\n\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

sess = tf.InteractiveSession()

# define placeholder
x = tf.placeholder(tf.float32, shape=[None, RESOLUTION, RESOLUTION, 3], name='x') 
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
    
# define inferences
net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)

# define cost function and metric.
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy') 

# add regularizer
n_params = len(net_train.all_params)
for i in range(n_params):
    cost = cost + tf.contrib.layers.l2_regularizer(0.2)(net_train.all_params[i])

# cost and accuracy for evaluation
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, LayersConfig.tf_dtype))

# define the optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 20000, 0.95, staircase=True)
train_params = net_train.all_params
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=train_params, global_step=global_step)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
log_string('NETWORK VARIABLES\n')
log_string(str(net_train.all_params)+'\n')
log_string('NETWORK LAYERS\n')
log_string(str(net_train.all_layers))
log_string('------------------------')

# configure dataset
## YOU NEED TO CONFIGURE THE PATH TO FIT YOURS DATASET ##
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, '2nd_test')
TEST_DIR = os.path.join(DATA_DIR, 'samples')

train_r_files = glob(os.path.join(TRAIN_DIR, '01_*', '*.*'))
train_g_files = glob(os.path.join(TRAIN_DIR, '00_pgan*jpg', "*.*"))

val_r_files = glob(os.path.join(VAL_DIR, '01_real*', "*.*"))
val_g_files = glob(os.path.join(VAL_DIR, '00_gen', '*', "*.*"))

num_batch_t = min(len(train_r_files), len(train_g_files)) // BATCH_SIZE*2

num_iter = 0
min_err = np.inf

if LOAD_TIME is not None:
    tl.files.load_and_assign_npz(sess=sess, name=os.path.join(CHECKPOINT_DIR, NPZ_TO_LOAD), network=net_train)
    num_iter = int(NPZ_TO_LOAD.split('_')[-2])

for epoch in range(num_iter // num_batch_t, NUM_EPOCH):

    shuffle(train_r_files)
    shuffle(train_g_files)

    train_loss, train_acc = 0, 0 
    for t_idx in range(0, num_batch_t):
        start_time = time.time()

        # load training data
        train_r_images = tl.visualize.read_images(train_r_files[t_idx*BATCH_SIZE//2:(t_idx+1)*BATCH_SIZE//2], printable=False)
        
        # Random zoom
        train_r_images = tl.prepro.threading_data(train_r_images, tl.prepro.imresize, size=[RESOLUTION*2, RESOLUTION*2])
        train_r_images = tl.prepro.threading_data(train_r_images, tl.prepro.zoom, zoom_range=(1.0, 1.1))
        train_r_images = tl.prepro.threading_data(train_r_images, tl.prepro.crop, wrg = RESOLUTION*2, hrg = RESOLUTION*2)
        train_r_images = tl.prepro.threading_data(train_r_images, tl.prepro.imresize, size=[RESOLUTION, RESOLUTION], interp=random.choice(['bicubic', 'bilinear']))

        train_g_images = tl.visualize.read_images(train_g_files[t_idx*BATCH_SIZE//2:(t_idx+1)*BATCH_SIZE//2], printable=False)

        train_g_images = tl.prepro.threading_data(train_g_images, tl.prepro.imresize, size=[RESOLUTION*2, RESOLUTION*2])
        train_g_images = tl.prepro.threading_data(train_g_images, tl.prepro.zoom, zoom_range=(1.0, 1.1))
        train_g_images = tl.prepro.threading_data(train_g_images, tl.prepro.crop, wrg = RESOLUTION*2, hrg = RESOLUTION*2)
        train_g_images = tl.prepro.threading_data(train_g_images, tl.prepro.imresize, size=[RESOLUTION, RESOLUTION], interp=random.choice(['bicubic', 'bilinear']))

        train_r_labels = np.zeros(BATCH_SIZE//2) 
        train_g_labels = np.ones(BATCH_SIZE//2) 

        X_train = np.concatenate([train_r_images, train_g_images])
        Y_train = np.concatenate([train_r_labels, train_g_labels])
        
        # load validation data
        batch_files = np.random.choice(val_r_files, BATCH_SIZE//2)
        val_r_images = tl.visualize.read_images(batch_files, printable=False)
        val_r_images = tl.prepro.threading_data(val_r_images, tl.prepro.imresize, size=[RESOLUTION, RESOLUTION], interp=random.choice(['bicubic', 'bilinear']))

        batch_files = np.random.choice(val_g_files, BATCH_SIZE//2)
        val_g_images = tl.visualize.read_images(batch_files, printable=False)
        val_g_images = tl.prepro.threading_data(val_g_images, tl.prepro.imresize, size=[RESOLUTION, RESOLUTION], interp=random.choice(['bicubic', 'bilinear']))      

        val_r_labels = np.zeros(len(val_r_images)) 
        val_g_labels = np.ones(len(val_g_images))

        X_val = np.concatenate([val_r_images, val_g_images])
        Y_val = np.concatenate([val_r_labels, val_g_labels])

        err, ac, _ = sess.run([cost_test, acc, train_op], feed_dict={x: X_train, y_: Y_train})
        train_loss += err
        train_acc += ac
        val_err, val_ac = sess.run([cost_test, acc], feed_dict={x: X_val, y_: Y_val})

        if num_iter % PRINT_FREQ == 0 and num_iter > 0:
            log_string("Epoch: [%2d/%2d] [%4d/%4d] (iter.: %5d) time: %4.4f, tr_loss: %.4f, tr_acc: %.2f, val_loss: %.4f, val_acc: %.2f"
                        % (epoch, NUM_EPOCH, t_idx, num_batch_t, num_iter, time.time() - start_time, err, ac*100, val_err, val_ac*100)) 

        num_iter += 1

    # Validation at the end of every epoch
    sets = ['REAL', 'GAN']
    VAL_BATCH_SIZE = 4
    total_v_loss, total_v_acc, total_v_batch = 0, 0, 0
    for label, val_files in enumerate([val_r_files, val_g_files]):
        num_batch_v = len(val_files) // VAL_BATCH_SIZE
        total_v_batch += num_batch_v

        val_loss, val_acc = 0, 0
        for v_idx in range(0, num_batch_v):
            val_images = tl.visualize.read_images(val_files[v_idx*VAL_BATCH_SIZE:(v_idx+1)*VAL_BATCH_SIZE], printable=False)
            val_images = tl.prepro.threading_data(val_images, tl.prepro.imresize, size=[RESOLUTION, RESOLUTION], interp='bilinear')
            val_labels = np.ones(VAL_BATCH_SIZE)*label

            X_val = val_images
            Y_val = val_labels
            
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val, y_: Y_val})

            val_loss += err
            total_v_loss += err
            val_acc += ac
            total_v_acc += ac

        val_loss, val_acc = val_loss / num_batch_v, val_acc / num_batch_v
        log_string("[*] End of Epoch: [%2d/%2d] %s val_loss: %.4f, val_acc: %.2f" % (epoch, NUM_EPOCH, sets[label], val_loss, val_acc*100))

    total_t_loss, total_t_acc = train_loss / num_batch_t, train_acc / num_batch_t
    total_v_loss, total_v_acc = total_v_loss / total_v_batch, total_v_acc / total_v_batch
    log_string("[*] End of Epoch: [%2d/%2d] time: %4.4f, val_loss: %.4f, val_acc: %.2f" % (epoch, NUM_EPOCH, time.time() - start_time, total_v_loss, total_v_acc*100))
    log_string("[*] End of Epoch: [%2d/%2d] time: %4.4f, train_loss: %.4f, train_acc: %.2f" % (epoch, NUM_EPOCH, time.time() - start_time, total_t_loss, total_t_acc*100))

    # if val_loss < min_err, save model
    if total_v_loss < min_err:    
        min_err = total_v_loss
        tl.files.save_npz(net_train.all_params, name=os.path.join(CHECKPOINT_DIR, now + '_' + str(epoch) + '_' + str(num_iter) + '_minErr.npz'), sess=sess)
        log_string("[*] Minimum validation error updated [%.4f]" % min_err)
        log_string("[*] Saving [%d epoch] checkpoints SUCCESS!" % epoch)

    if epoch % SAVE_INTERVAL == 0 and epoch is not 0:
        tl.files.save_npz(net_train.all_params, name=os.path.join(CHECKPOINT_DIR, now + '_' + str(epoch) + '_' + str(num_iter) + '_net.npz'), sess=sess)
        log_string("[*] Saving [%d epoch] checkpoints SUCCESS!" % epoch)
