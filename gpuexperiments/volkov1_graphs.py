from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import array
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
Y2_list = []
Y3_list = []
Y4_list = []
Y6_list = []
Y8_list = []
x_values = set()

x_values = set()
for timeinfo in times:
    x = int(timeinfo['name'].split('_')[-1])
    x_values.add(x)
x_list = sorted(list(x_values))
pos_by_x = {}
for i, x in enumerate(x_list):
    pos_by_x[x] = i

X_list = []
X2_list = []
X3_list = []
X4_list = []
X6_list = []
X8_list = []
Y_list = []
Y2_list = []
Y3_list = []
Y4_list = []
Y6_list = []
Y8_list = []
for timeinfo in times:
    name = timeinfo['name']
    x = int(timeinfo['name'].split('_')[-1])
    if name.startswith('k1_fma_') and not name.startswith('k1_fma_ilp'):
        X_list.append(x)
        Y_list.append(timeinfo['flops'])
    elif name.startswith('k1_fma_ilp2'):
        X2_list.append(x)
        Y2_list.append(timeinfo['flops'])
    elif name.startswith('k1_fma_ilp3'):
        X3_list.append(x)
        Y3_list.append(timeinfo['flops'])
    elif name.startswith('k1_fma_ilp4'):
        X4_list.append(x)
        Y4_list.append(timeinfo['flops'])
    elif name.startswith('k1_fma_ilp6'):
        X6_list.append(x)
        Y6_list.append(timeinfo['flops'])
    elif name.startswith('k1_fma_ilp8'):
        X8_list.append(x)
        Y8_list.append(timeinfo['flops'])
X = np.array(X_list)
X2 = np.array(X2_list)
X3 = np.array(X3_list)
X4 = np.array(X4_list)
X6 = np.array(X6_list)
X8 = np.array(X8_list)
Y = np.array(Y_list)
Y2 = np.array(Y2_list)
Y3 = np.array(Y3_list)
Y4 = np.array(Y4_list)
Y6 = np.array(Y6_list)
Y8 = np.array(Y8_list)

thismax = 0
plt.plot(X, Y, label='ilp 1')
thismax = max(thismax, max(Y))
plt.plot(X2, Y2, label='ilp 2')
thismax = max(thismax, max(Y2))
plt.plot(X3, Y3, label='ilp 3')
thismax = max(thismax, max(Y3))
if len(Y4_list) > 0:
    plt.plot(X4, Y4, label='ilp 4')
    thismax = max(thismax, max(Y4))
if len(Y6_list) > 0:
    plt.plot(X6, Y6, label='ilp 6')
    thismax = max(thismax, max(Y6))
if len(Y8_list) > 0:
    plt.plot(X8, Y8, label='ilp 8')
    thismax = max(thismax, max(Y8))

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(X), 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Blocksize')
plt.ylabel('GFLOPS')
legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/volkov1_%s.png' % deviceNameSimple, dpi=150)

