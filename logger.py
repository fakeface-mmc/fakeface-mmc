import tensorflow as tf
from datetime import datetime


class Logger(object):
    def __init__(self, log_path, tb_path):
        self.LOG_FOUT = open(log_path, 'a')
        self.LOG_FOUT.write('****' + datetime.now().strftime(r'%y-%m-%d(%Hh%Mm)') + '****' + '\n')

        self.writer = tf.summary.FileWriter(tb_path)

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
