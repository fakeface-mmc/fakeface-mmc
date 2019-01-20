import os
from glob import glob
import numpy as np
import tensorlayer as tl
from predict_M1.model import *
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flags = tf.app.flags
flags.DEFINE_integer("image_size", 64, "The size of the images [64]")
flags.DEFINE_string("load_dir", "pretrained", "Directory name containing .npz [checkpoint]")
flags.DEFINE_string("load_npz", "net.npz", "The name of npz to load [net]")
FLAGS = flags.FLAGS


def predict(data_dir):
    IMAGE_SIZE = FLAGS.image_size
    LOAD_NPZ = '19-01-06(17h31m)_12_32864_minErr.npz'
    cur_path = os.path.dirname( os.path.abspath( __file__ ))
    LOAD_DIR = FLAGS.load_dir
    LOAD_DIR = os.path.join(cur_path, LOAD_DIR)
    net_name = os.path.join(cur_path, LOAD_DIR, LOAD_NPZ)

    if tl.files.folder_exists(LOAD_DIR) is False:
        raise ValueError("checkpoint_dir {} does not exist.".format(LOAD_DIR))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tl.layers.initialize_global_variables(sess)

        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='x')
        y = tf.placeholder(tf.int64, [None], name='y_')

        net_train = model(x, is_train=True, reuse=False)
        net = model(x, is_train=False, reuse=True)
        y_ = net.outputs

        if tl.files.load_and_assign_npz(sess=sess, name=net_name, network=net_train) is False:
            print("[*] Loading checkpoints FAILURE!")
            exit(1)
        else:
            print("[*] Loading checkpoints SUCCESS!")

        data_files = sorted(glob(os.path.join(data_dir, "*.jpg")))
        f = open('gan_vs_others.txt', 'w')
        for i in range(len(data_files)):
            images = tl.visualize.read_image(data_files[i])
            filename = os.path.splitext(os.path.basename(data_files[i]))[0]

            images = images[np.newaxis,:]
            images = tl.prepro.threading_data(images, tl.prepro.imresize, size=[IMAGE_SIZE, IMAGE_SIZE])
            feed_dict = {x: images}
            pred = sess.run(tf.nn.softmax(y_), feed_dict=feed_dict)
            pred = np.squeeze(pred)
            data = "%s,%.4f\n" % (filename, pred[1])
            f.write(data)
        f.close()
