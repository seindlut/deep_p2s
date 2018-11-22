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