from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import csv
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
f = open('results/maths2_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    print('row', row)
    if row['name'] not in ['int_sub', 'int_add']:
        times.append({'name': row['name'], 'time': float(row['tot ms']), 'gflops': float(row['gflops'])})
#f.readline()
#for line in f:
    #split_line = line.split('\t')
    #times.append({'name': split_line[0], 'time': float(split_line[1]), 'flops': float(split_line[2])})
f.close()

print('times', times)
# sys.exit(1)

labels = []
values = []

for timeinfo in times:
    labels.append(timeinfo['name'].replace('_', ' '))
    values.append(timeinfo['gflops'])

y_pos = np.arange(len(labels))

# plt.plot(X, Y, label='ilp 1')

plt.barh(y_pos, values, align='center')

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(values), -0.5, len(values)])
plt.yticks(y_pos, labels)
plt.title(deviceNameSimple)
plt.xlabel('FLOPS')
#plt.ylabel('GFLOPS')
#legend = plt.legend(loc='lower right') # fontsize='x-large')
plt.savefig('/tmp/maths2_%s.png' % deviceNameSimple, dpi=150)

