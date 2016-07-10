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

X_list_by_ilp = defaultdict(list)
Y_list_by_ilp = defaultdict(list)

for timeinfo in times:
    name = timeinfo['name']
    x = int(timeinfo['name'].split('_')[-1].replace('bsm', ''))
    y = timeinfo['bw']
    ilp = 1
    if 'ilp' in name:
        ilp = int(timeinfo['name'].split('_')[1].replace('ilp', ''))
    X_list_by_ilp[ilp].append(x)
    Y_list_by_ilp[ilp].append(y)

thismax = 0
for ilp in sorted(X_list_by_ilp.keys()):
    X = np.array(X_list_by_ilp[ilp])
    Y = np.array(Y_list_by_ilp[ilp])
    plt.plot(X, Y, label='ilp %s' % ilp)
    thismax = max(thismax, max(Y))

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(X), 0, thismax])
plt.title(deviceNameSimple)
plt.xlabel('Blocksize')
plt.ylabel('Bandwidth (GiB)')
legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/volkov_memcpy_%s.png' % deviceNameSimple, dpi=150)

