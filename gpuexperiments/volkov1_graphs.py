from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib.pyplot as plt
from os.path import join
import lib_clgpuexp


parser = argparse.ArgumentParser()
parser.add_argument('--devicename')
args = parser.parse_args()

assert args.devicename is not None

times = []

deviceNameSimple = args.devicename
f = open('results/volkov1_%s.tsv' % args.devicename, 'r')
f.readline()
for line in f:
    split_line = line.split('\t')
    times.append({'name': split_line[0], 'time': float(split_line[1]), 'flops': float(split_line[2])})
f.close()

X_list = []
Y_list = []
for timeinfo in times:
    name = timeinfo['name']
    if not name.startswith('k1_fma_') or name.startswith('k1_fma_ilp'):
        continue
    X_list.append(int(name.split('fma_')[1].split(' ')[0]))
    Y_list.append(timeinfo['flops'])
X = np.array(X_list)
Y = np.array(Y_list)
plt.plot(X, Y)
plt.axis([0, max(X), 0, max(Y)])
plt.title(deviceNameSimple)
plt.xlabel('Blocksize')
plt.ylabel('GFLOPS')
plt.savefig('/tmp/volkov1_%s.png' % deviceNameSimple, dpi=150)

