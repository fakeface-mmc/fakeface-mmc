import os.path
import sys
import tensorflow as tf
import multiprocessing
from model.net_xception import model
from ntpath import basename
from tensorflow.contrib import slim
sum_interval_loss = 0
start_learning_rate = 5e-05
lr_update_interval = 10000
lr_update_rate = 0.95
beta = 0.0001  # l2 normalizer parameter(preventing overfitting)
MAX_EPOCH = 5000
batch_seen=0
loss_interval = 100
batch_size_ = 150
LOG_FOUT_train = open('./log/log_train.txt', "a")
is_train = True

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()

def parse_real_fn(x):
    img = tf.cast(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), dct_method='INTEGER_ACCURATE', channels=3), [64, 64]), dtype=tf.float32)
    img = tf.divide(img, 255)
    label = tf.constant([0,1], dtype= tf.float32)

    return img, label, x


def parse_syn_fn(x):
    img = tf.cast(
        tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), dct_method='INTEGER_ACCURATE', channels=3),
                                 [64, 64]), dtype=tf.float32)
    img = tf.divide(img, 255)
    label = tf.constant([1, 0], dtype=tf.float32)

    return img, label, x


def operation_batch_seen(train_loss_val, summary, curr_batch):
    global batch_seen
    global sum_interval_loss

    # print avg interval loss
    batch_seen += 1
    sum_interval_loss += train_loss_val
    if batch_seen % loss_interval == 0 and batch_seen > 0:
        mean_loss = sum_interval_loss / loss_interval
        sum_interval_loss = 0
        log_string(LOG_FOUT_train, "batch seen: %d, curr batch : %d,  interval train loss: %s, interval: %d" % (batch_seen, curr_batch, mean_loss, loss_interval))


def main():
    # data processing using tensorflow.Dataset api
    max_cpus = multiprocessing.cpu_count()
    real_filenames = tf.placeholder(tf.string)
    syn_filenames = tf.placeholder(tf.string)
    batch_size = tf.placeholder(tf.int64)

    real_files = tf.data.Dataset.list_files(real_filenames)
    real_dataset = real_files.map(parse_real_fn,num_parallel_calls=int(max_cpus / 2))

    syn_files = tf.data.Dataset.list_files(syn_filenames)
    syn_dataset = syn_files.map(parse_syn_fn, num_parallel_calls=int(max_cpus / 2))

    real_dataset = real_dataset.batch(batch_size)
    real_dataset = real_dataset.prefetch(buffer_size=200)
    real_dataset = real_dataset.shuffle(buffer_size=100)
    real_iterator = real_dataset.make_initializable_iterator()
    real_img, real_label, real_name = real_iterator.get_next()

    syn_dataset = syn_dataset.batch(batch_size)
    syn_dataset = syn_dataset.prefetch(buffer_size=200)
    syn_dataset = syn_dataset.shuffle(buffer_size=100)
    syn_iterator = syn_dataset.make_initializable_iterator()
    syn_img, syn_label, syn_name = syn_iterator.get_next()

    # Tensor batch imgs, labels and filnames
    img = tf.concat([real_img, syn_img], axis=0)
    label = tf.concat([real_label, syn_label], axis=0)
    name = tf.concat([real_name, syn_name], axis=0)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        #phase for batch normalization. True for training, False for test
        phase = tf.placeholder(tf.bool, name='phase')
        # Model initialize
        ce, acc, merge, prediction = model(img, label, phase)
        # Tensorboard instance
        writer = tf.summary.FileWriter('./log', sess.graph)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, lr_update_interval, lr_update_rate,
                                                   staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Define training operation
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum = 0.9)
            train_op = slim.learning.create_train_op(ce, optimizer)

        tf.global_variables_initializer().run()


        cur_path = os.path.dirname(os.path.abspath(__file__))
        saver = tf.train.Saver(max_to_keep=300)

        #modify dat_dir to your data files
        data_dir = './../data/zoomed/'
        trainig_filenames = data_dir + 'train/'
        test_filenames = data_dir + 'test/'


        if is_train:
            for epoch in range(0, MAX_EPOCH):

                #Initialize training dataset. Modify real_filenames and syn_filenaes to your data
                sess.run([real_iterator.initializer, syn_iterator.initializer], feed_dict={ real_filenames : trainig_filenames + 'real/*/*.jpg', syn_filenames : trainig_filenames +'syn/*/*.jpg', batch_size :  batch_size_})

                log_string(LOG_FOUT_train, '**** Train EPOCH {} ****'.format(epoch))
                sys.stdout.flush()
                loss_sum = 0
                curr_batch=0

                #Train until the tf.Dataset API iterates whole dataset once
                while True:
                    try:
                        _, train_loss_val, summary = sess.run([train_op, ce, merge], feed_dict={phase: True})
                        writer.add_summary(summary, batch_seen)
                        curr_batch+=1
                        loss_sum += train_loss_val
                        operation_batch_seen(train_loss_val, summary, curr_batch)

                    except tf.errors.OutOfRangeError:
                        break

                # Save the variables to disk.
                saver.save(sess, os.path.join('./log', 'epoch' + str(epoch) + '_model.ckpt'))
                log_string(LOG_FOUT_train, 'mean loss : {}, batch seen: {}'.format(loss_sum / float(curr_batch), batch_seen))


                # Initialize test dataset
                sess.run([real_iterator.initializer, syn_iterator.initializer],
                         feed_dict={real_filenames: test_filenames + 'real/*.jpg',
                                    syn_filenames: test_filenames + 'syn/*.jpg', batch_size: 40})

                LOG_FOUT_test = open('./log/log_test{}.txt'.format(epoch), "a")

                log_string(LOG_FOUT_test, 'fname prediction')
                sum_acc = 0
                sum_step = 0
                loss_sum = 0
                # Test images with currently trained model
                while True:
                    try:
                        temp_accuracy, temp_prediction, fname, label_,  test_loss_val= sess.run(
                            [acc, prediction, name, label, ce], feed_dict={phase: False})
                        sum_acc += temp_accuracy
                        sum_step += 1
                        loss_sum += test_loss_val
                        for i in range(len(fname)):
                            name_ = basename(tf.compat.as_text(fname[i], encoding='utf-8'))
                            name_ = name_.split('.')[0]
                            log_string(LOG_FOUT_test, name_ + ' {0:.3f}'.format(temp_prediction[i][0]))

                    except tf.errors.OutOfRangeError:
                        break
                summary = tf.Summary()
                summary.value.add(tag='Accuracy', simple_value=sum_acc/sum_step)
                summary.value.add(tag='TestLoss', simple_value = loss_sum / sum_step)
                writer.add_summary(summary, epoch)

                log_string(LOG_FOUT_test, 'accuracy : {}'.format(sum_acc/sum_step))
                log_string(LOG_FOUT_train, 'accuracy : {}'.format(sum_acc/sum_step))

if __name__=='__main__':
    main()