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
f = open('results/globalwrite_%s.tsv' % args.devicename, 'r')
reader = csv.DictReader(f, delimiter='\t')
for row in reader:
    times.append({'name': row['name'], 'bw': float(row['bw gib'])})
f.close()

def getNameValue(name, key, default=None):
    for bit in name.split('_'):
        if bit.startswith(key):
            return int(bit.replace(key, ''))
    if default is not None:
        return default
    raise Exception('key %s not found in %s' % (key, name))

def varyilp():
    X_lists = []
    Y_lists = []
    names = []

    lastFamily = None
    list_idx = -1
    for timeinfo in times:
        name = timeinfo['name']
        family = name.split('bsm')[0]
        layout = getNameValue(name, 'l', 1)
        stride = getNameValue(name, 'stride', 1)
        if stride != 1:
            continue
        if 'ilp' in name:
            ilp = getNameValue(timeinfo['name'], 'ilp', 1)
            stride = getNameValue(timeinfo['name'], 'stride', 1)
            # ilp = int(timeinfo['name'].split('_')[1].replace('ilp', ''))
        layout = getNameValue(timeinfo['name'], 'l', 1)
        if layout != 1:
            continue
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
    maxx = 0
    for i, name in enumerate(names):
        X = np.array(X_lists[i])
        Y = np.array(Y_lists[i])
        plt.plot(X, Y, label=name)
        thismax = max(thismax, max(Y))
        maxx = max(maxx, max(X))
    thismax *= 1.05

    #plt.axis([0, max(X), 0, max(Y)])
    plt.axis([0, maxx, 0, thismax])
    plt.title(deviceNameSimple)
    plt.xlabel('Blocks per SM')
    plt.ylabel('Bandwidth (GiB/s)')
    legend = plt.legend(loc='lower right') # fontsize='x-large')
    plt.savefig('/tmp/globalwrite_varyilp_%s.png' % deviceNameSimple, dpi=150)
    plt.close()

def varystride():
    X_lists = []
    Y_lists = []
    names = []

    lastFamily = None
    list_idx = -1
    for timeinfo in times:
        name = timeinfo['name']
        family = name.split('bsm')[0]
        layout = getNameValue(name, 'l', 1)
        ilp = int(getNameValue(name, 'ilp', 1))
        # print('ilp', ilp)
        if ilp != 1:
            continue
        if layout != 1:
            continue
        # print('ilp', ilp)
        stride = int(getNameValue(name, 'stride', 1))
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
            names.append('stride %s' % (stride,))
        x = int(timeinfo['name'].split('_')[-1].replace('bsm', ''))
        y = timeinfo['bw']
        X_lists[list_idx].append(x)
        Y_lists[list_idx].append(y)

    thismax = 0
    maxx = 0
    for i, name in enumerate(names):
        X = np.array(X_lists[i])
        Y = np.array(Y_lists[i])
        plt.plot(X, Y, 'x-', label=name)
        thismax = max(thismax, max(Y))
        maxx = max(maxx, max(X))
    thismax *= 1.05

    #plt.axis([0, max(X), 0, max(Y)])
    plt.axis([0, maxx, 0, thismax])
    plt.title(deviceNameSimple)
    plt.xlabel('Blocks per SM')
    plt.ylabel('Bandwidth (GiB/s)')
    legend = plt.legend(loc='lower right') # fontsize='x-large')
    plt.savefig('/tmp/globalwrite_varystride_%s.png' % deviceNameSimple, dpi=150)
    plt.close()

varyilp()
varystride()

