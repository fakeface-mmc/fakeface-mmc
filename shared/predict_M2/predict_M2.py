import os.path
import tensorflow as tf
import multiprocessing
from predict_M2.model.net_xception import model_predict
import glob
from ntpath import basename
from predict_M2.utils.face_crop import crop
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()

def parse_fn(img, x):
    img = tf.cast(
        tf.image.resize_images(tf.image.decode_jpeg(img, dct_method='INTEGER_ACCURATE', channels=3),
                               [64, 64]), dtype=tf.float32)
    image_resized = tf.divide(img, 255)
    return image_resized, x

def predict(data_dir):
    data_dir = data_dir+'/*.jpg'

    max_cpus = multiprocessing.cpu_count()
    batch_size = tf.placeholder(tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices(glob.glob(data_dir))
    dataset = dataset.map(lambda filename : tuple(tf.py_func(crop,[filename], [tf.string, tf.string])), num_parallel_calls=int(max_cpus / 2))
    dataset = dataset.map(parse_fn, num_parallel_calls=int(max_cpus / 2))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=200)
    iterator = dataset.make_initializable_iterator()
    img, name = iterator.get_next()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        phase = tf.placeholder(tf.bool, name='phase')
        prediction = model_predict(img, phase)

        tf.global_variables_initializer().run()

        cur_path = os.path.dirname(os.path.abspath(__file__))


        restore_epoch = 12
        saver = tf.train.Saver(max_to_keep=300, var_list=[v for v in tf.global_variables() if v.name.startswith('block')])
        ckpt_dir = os.path.join(cur_path, 'log', 'epoch' + str(restore_epoch) + '_model.ckpt')
        saver.restore(sess, ckpt_dir)
        LOG_FOUT_test = open('./syn_vs_others.txt'.format(restore_epoch), "w")
        sess.run(iterator.initializer, feed_dict={batch_size: 4})

        while True:
            try:

                prediction_, fname = sess.run([prediction, name], feed_dict={phase: False})
                for i in range(len(fname)):
                    name_ = basename(tf.compat.as_text(fname[i], encoding='utf-8'))
                    name_ = name_.split('.')[0]
                    log_string(LOG_FOUT_test, name_ + ' {0:.3f}'.format(1 - prediction_[i][0]))


            except tf.errors.OutOfRangeError:
                break


