from __future__ import print_function, division
import argparse
import string
import numpy as np
import os
import csv
import array
import matplotlib as mp
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib.pyplot as plt
from os.path import join
import lib_clgpuexp


parser = argparse.ArgumentParser()
parser.add_argument('--devicename', required=True)
args = parser.parse_args()

times = []

deviceNameSimple = args.devicename
f = open('results/inline_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    print('row', row)
    name = row['name']
    times.append({'name': name, 'time': float(row['tot ms']), 'gflops': float(row['gflops'])})
f.close()

labels = []
values = []

times.reverse()
for timeinfo in times:
    #labels.append(timeinfo['name'].replace('_128', '').replace('noprag_', 'unroll_').replace('noopt', '').replace('k1_', '').replace('_', ' ').strip())
    labels.append(timeinfo['name'].replace('k_', '').replace('_', ' '))
    values.append(timeinfo['gflops'])

y_pos = np.arange(len(labels))

# plt.plot(X, Y, label='ilp 1')

plt.barh(y_pos, values, align='center')

#plt.axis([0, max(X), 0, max(Y)])
plt.axis([0, max(values), -0.5, len(values)])
plt.yticks(y_pos, labels, fontproperties=mp.font_manager.FontProperties(size=8))
plt.title(deviceNameSimple)
plt.xlabel('GFLOPS/second', fontproperties=mp.font_manager.FontProperties(size=10))
#plt.tick_params(axis='y', pad=-80, direction='in', left='off')
#plt.ylabel('GFLOPS')
#legend = plt.legend(loc='lower right') # fontsize='x-large')
#plt.yticks(y_pos, people, fontproperties=font_manager.FontProperties(size=8))
plt.savefig('/tmp/inline_%s.png' % deviceNameSimple, dpi=150)

