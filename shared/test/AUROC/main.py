#!/usr/bin/python
'''
	File name : main.py
	Author : Hyoungseok Chu (hyoungseok.chu@gmail.com)
	Date created : 18/09/2017
	Date last modified : 15/09/2017
	Python version : 2.7.5
'''

import sys
import argparse
from auroc import compute_auroc, read_file, read_merged_file

"""
Arguments rule
-p : prediction file name
-l : label file name
-i : merged (problem number, prediction, label order) file name

ex) $ python ./main.py -p prediction_file.txt -l label_file.txt
    $ python ./main.py -i merged_pl.txt
"""
argparser = argparse.ArgumentParser(description='compute AUROC to evaluate binary classification performance.')
argparser.add_argument('-p', '--pred', default='prediction_file.txt', help='prediction result file.')
argparser.add_argument('-l', '--label', default='label_file.txt', help='label file.')
argparser.add_argument('-i', '--integrated', default='', help='prediction+label merged file. The input file must consist of triples of ID,pred_prob,and label. DO NOT USE WITH -p(--pred) OR -l(--label) OPTIONS. if -i option is assigned, -p and -l options will be ignored.')

args = argparser.parse_args()

predict_file = args.pred 
label_file   = args.label

[predict, target] = read_file(predict_file, label_file)

[auc, roc] = compute_auroc(predict,target)

sys.stderr.write('Computed AUROC score: ')
sys.stderr.flush()

print('%0.9f' % (auc))
