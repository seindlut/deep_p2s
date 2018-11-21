from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

import os
import os.path as osp

# add for print log
import datetime
import sys
from tee_print import Tee
try:
    from tabulate import tabulate
except:
    print "tabulate lib not installed"


def print_log(log_prefix, FLAGS):
    # log direcory and file
    net_name = 'Sketch-RNN-SBIR-Net'
    log_file = log_prefix + os.uname()[1].split('.')[0] + '_' + datetime.datetime.now().isoformat().split('.')[0].replace('-','_').replace(':', '_') + '_log.txt'
    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    print "logging to ", log_file, "..."

    # print header
    print "==============================================="
    print "Trainning ", net_name, " in this framework"
    print "==============================================="

    print "Tensorflow flags:"

    flags_list = []
    for attr, value in sorted(FLAGS.__flags.items()):
        flags_list.append(attr)
    FLAGS.saved_flags = " ".join(flags_list)

    flag_table = {}
    flag_table['FLAG_NAME'] = []
    flag_table['Value'] = []
    flag_lists = FLAGS.saved_flags.split()
    # print self.FLAGS.__flags
    for attr in flag_lists:
        if attr not in ['saved_flags', 'net_name', 'log_root']:
            flag_table['FLAG_NAME'].append(attr.upper())
            flag_table['Value'].append(getattr(FLAGS, attr))
    flag_table['FLAG_NAME'].append('NET_NAME')
    flag_table['Value'].append(net_name)
    flag_table['FLAG_NAME'].append('HOST_NAME')
    flag_table['Value'].append(os.uname()[1].split('.')[0])
    try:
        print tabulate(flag_table, headers="keys", tablefmt="fancy_grid").encode('utf-8')
    except:
        for attr in flag_lists:
            print "attr name, ", attr.upper()
            print "attr value, ", getattr(FLAGS, attr)


def test_reshape():
    ndim = [2, 3, 4]
    nlen = ndim[0] * ndim[1] * ndim[2]
    a_origin = np.arange(nlen)
    a1 = np.reshape(a_origin, ndim)
    a2 = np.reshape(a1, [-1, ndim[2]])
    a3 = np.reshape(a2, ndim)
    print a1 - a3


if __name__ == '__main__':
    # sketch_category = 'aaron_sheep'
    test_reshape()