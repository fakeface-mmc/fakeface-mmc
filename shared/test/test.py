from predict_M1 import  predict_M1
from predict_M2 import predict_M2
import tensorflow as tf
import os
from AUROC.auroc import compute_auroc, read_file
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flags = tf.app.flags
flags.DEFINE_string("data_dir", r"D:\Download\real", "data_dir")
flags.DEFINE_float("threshold", 0.9, "gan threshold")
flags.DEFINE_string("output", "result.txt", "output name")

FLAGS = flags.FLAGS



def match_name(lines, key):
    for line in lines:
        if line.strip().split()[0] == key:
            return line.strip().split()
    return None

def merge(threshold, out_name):
    f1_name = 'gan_vs_others.txt'
    f2_name = 'syn_vs_others.txt'

    f1 = open(f1_name, 'r')
    f2 = open(f2_name, 'r')
    lines1 = f1.readlines()
    lines2 = f2.readlines()

    num_of_lines = len(lines1)

    f3 = open(out_name, 'w')

    for idx in range(num_of_lines):

        line1 = lines1[idx].strip().split(',')

        input_line = line1[0] + ","

        if float(line1[1]) <= threshold:
            line2 = match_name(lines2, line1[0])
            if line2 != None:
                input_line += line2[1]
            else:
                print("what is this file: %s" % line1[0])
                input_line += line1[1]
        else:
            input_line += line1[1]
        input_line += "\n"
        f3.write(input_line)
    f3.close()
    f1.close()
    f2.close()
    os.remove(f1_name)
    os.remove(f2_name)



def main():
    data_dir = FLAGS.data_dir
    threshold = FLAGS.threshold
    # Test images in data_dir with GAN network
    predict_M1.predict(data_dir)
    # Test images in data_dir with SYN network
    predict_M2.predict(data_dir)
    max = -1

    # With given threshold, if the output of GAN network is higher than the threshold, set the forgery probability as GAN network output. Otherwise, the forgery probability is same as SYN network
    merge(threshold=threshold, out_name=FLAGS.output)
    [predict, target] = read_file('result.txt', 'gt.txt')
    [auc, roc] = compute_auroc(predict,target)
    print('Threshold: %.3f AUROC:%0.9f' % (threshold, auc))



if __name__ == '__main__':
    main()

