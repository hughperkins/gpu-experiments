"""
Try using dynamic shared memory, see if gets optimized away, or affects occupancy
"""
from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib.pyplot as plt
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--devicename')
args = parser.parse_args()

times = []
assert args.devicename is not None
deviceNameSimple = args.devicename
f = open('results/occupancy_dyn_%s.tsv' % args.devicename, 'r')
f.readline()
for line in f:
    split_line = line.split('\t')
    times.append({'name': split_line[0], 'time': float(split_line[1]), 'flops': float(split_line[2])})
f.close()

X32_list = []
Y32_list = []
X64_list = []
Y64_list = []
for timeinfo in times:
    name = timeinfo['name']
    if not name.startswith('k1_g1024_b'):
        continue
    block = int(name.split('_')[2].replace('b', ''))
    x = int(name.split('_')[-1].replace('s', ''))
    y = timeinfo['flops']
    if block == 32:
        X32_list.append(x)
        Y32_list.append(y)
    elif block == 64:
        X64_list.append(x)
        Y64_list.append(y)
X32 = np.array(X32_list)
X64 = np.array(X64_list)
Y32 = np.array(Y32_list)
Y64 = np.array(Y64_list)
plt.plot(X32, Y32, label='blocksize 32')
plt.plot(X64, Y64, label='blocksize 64')
plt.axis([0, max(X32), 0, max(Y64)])
plt.title(deviceNameSimple)
plt.xlabel('Shared memory per block (KiB)')
plt.ylabel('GFLOPS')
legend = plt.legend(loc='upper right') # fontsize='x-large')
plt.savefig('/tmp/occupancy_by_shared_%s.png' % deviceNameSimple, dpi=150)
plt.close()

X_list = []
Y_list = []
for timeinfo in times:
    name = timeinfo['name']
    if not name.startswith('kernel_bsm'):
        continue
    X_list.append(int(name.split('bsm')[1].split(' ')[0]))
    Y_list.append(timeinfo['flops'])
X = np.array(X_list)
Y = np.array(Y_list)
plt.plot(X, Y)
plt.axis([0, max(X), 0, max(Y)])
plt.title(deviceNameSimple)
plt.xlabel('blocks per SM')
plt.ylabel('GFLOPS')
plt.savefig('/tmp/occupancy_%s.png' % deviceNameSimple, dpi=150)

