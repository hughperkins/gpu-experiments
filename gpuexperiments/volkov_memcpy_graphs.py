from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
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
f = open('results/volkov_memcpy_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    times.append({'name': row['name'], 'bw': float(row['bw gib'])})
f.close()

X_list = []
Y_list = []
X2_list = []
Y2_list = []
for timeinfo in times:
    name = timeinfo['name']
    x = int(timeinfo['name'].split('_')[-1].replace('bsm', ''))
    y = timeinfo['bw']
    if name.startswith('memcpy_bsm') and 'ilp' not in name:
        X_list.append(x)
        Y_list.append(y)
    else:
        ilp = int(timeinfo['name'].split('_')[1].replace('ilp', ''))
        if ilp == 2:
            X2_list.append(x)
            Y2_list.append(y)
X = np.array(X_list)
Y = np.array(Y_list)
X2 = np.array(X2_list)
Y2 = np.array(Y2_list)

# print('X', X)
# print('Y', Y)
thismax = 0
plt.plot(X, Y, label='ilp 1')
thismax = max(thismax, max(Y))
plt.plot(X2, Y2, label='ilp 2')
thismax = max(thismax, max(Y2))

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(X), 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Blocksize')
plt.ylabel('Bandwidth (GiB)')
legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/volkov_memcpy_%s.png' % deviceNameSimple, dpi=150)

