from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import csv
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
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    times.append({'name': row['name'], 'time': float(row['tot ms']), 'flops': float(row['gflops'])})
f.close()

names = []
X_lists = []
Y_lists = []

list_idx = -1
lastFamily = None
for timeinfo in times:
    name = timeinfo['name']
    family = '_'.join(name.split('_')[:3])
    ilp = int(name.split('ilp')[1].split('_')[0])
    if family != lastFamily:
        lastFamily = family
        list_idx += 1
        X_lists.append([])
        Y_lists.append([])
        names.append('ilp %s' % ilp)
    x = int(timeinfo['name'].split('_')[-1])
    y = timeinfo['flops']
    X_lists[list_idx].append(x)
    Y_lists[list_idx].append(y)

thismax = 0
for i, name in enumerate(names):
    X = np.array(X_lists[i])
    Y = np.array(Y_lists[i])
    plt.plot(X, Y, label=name)
    thismax = max(thismax, max(Y))

plt.axis([0, max(X), 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Blocksize')
plt.ylabel('GFLOPS')
legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/volkov1_%s.png' % deviceNameSimple, dpi=150)

