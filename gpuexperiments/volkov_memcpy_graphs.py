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
f = open('results/volkov_memcpy_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    times.append({'name': row['name'], 'bw': float(row['bw gib'])})
f.close()

X_lists = []
Y_lists = []
names = []
#X_list_by_ilp = defaultdict(list)
#Y_list_by_ilp = defaultdict(list)

lastFamily = None
list_idx = -1
for timeinfo in times:
    name = timeinfo['name']
    family = name.split('bsm')[0]
    ilp = 1
    if 'ilp' in name:
        ilp = int(timeinfo['name'].split('_')[1].replace('ilp', ''))
    if family != lastFamily:
        list_idx += 1
        X_lists.append([])
        Y_lists.append([])
        lastFamily = family
        floatType = 'float'
        if 'float2' in family:
            floatType = 'float2'
        if 'float4' in family:
            floatType = 'float4'
        names.append('%s ilp %s' % (floatType, ilp))
    x = int(timeinfo['name'].split('_')[-1].replace('bsm', ''))
    y = timeinfo['bw']
    X_lists[list_idx].append(x)
    Y_lists[list_idx].append(y)

thismax = 0
for i, name in enumerate(names):
    X = np.array(X_lists[i])
    Y = np.array(Y_lists[i])
    plt.plot(X, Y, label=name)
    thismax = max(thismax, max(Y))

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(X), 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Blocks per SM')
plt.ylabel('Bandwidth (GiB/s)')
legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/volkov_memcpy_%s.png' % deviceNameSimple, dpi=150)

