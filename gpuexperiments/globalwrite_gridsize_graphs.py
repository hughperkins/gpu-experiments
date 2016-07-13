from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
from collections import defaultdict
import array
import csv
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib.pyplot as plt
from os.path import join
import lib_clgpuexp


parser = argparse.ArgumentParser()
parser.add_argument('--devicename', required=True)
args = parser.parse_args()

assert args.devicename is not None

times = []

deviceNameSimple = args.devicename
f = open('results/globalwrite_gridsize_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    times.append({'name': row['name'], 'bw': float(row['bw gib'])})
f.close()

X_lists = []
Y_lists = []
names = []

def getNameValue(name, key, default=None):
    for bit in name.split('_'):
        if bit.startswith(key):
            return int(bit.replace(key, ''))
    if default is not None:
        return default
    raise Exception('key %s not found in %s' % (key, name))

lastFamily = None
list_idx = -1
for timeinfo in times:
    name = timeinfo['name']
    family = 'a'
    blockSize = int(name.replace('gridsize', ''))
    if family != lastFamily:
        list_idx += 1
        X_lists.append([])
        Y_lists.append([])
        lastFamily = family
        names.append('gridsize 1')
    x = blockSize
    y = timeinfo['bw']
    X_lists[list_idx].append(x)
    Y_lists[list_idx].append(y)

thismax = 0
maxx = 0
for i, name in enumerate(names):
    X = np.array(X_lists[i])
    Y = np.array(Y_lists[i])
    plt.plot(X, Y, '-x', label=name)
    thismax = max(thismax, max(Y))
    maxx = max(maxx, max(X))
if args.devicename == '940m':
    maxx = 48

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, maxx, 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Gridsize')
plt.ylabel('Bandwidth (GiB/s)')
# legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/globalwrite_gridsize_%s.png' % deviceNameSimple, dpi=150)

