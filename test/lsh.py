#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing, time
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser(description="Build locality sensitve hash database for 80M images")
#parser.add_argument('--db', required = True, help='Path to the tiny images binary')
parser.add_argument('-s', required = True, help='Hash Size')
rp = parser.parse_args()
#f = open(rp.db)
d = 28


start = time.time()

# -------------------------------------------------------------------
#def compute_distance(datachunks):
#	return [np.linalg.norm(qi - np.fromstring(c, np.uint8)) for c in datachunks]

def compute_hash(sample, uniform_planes):
    return ["".join(['1' if i > 0 else '0' for i in np.dot(p, (sample-np.full(d*d*1,128)).transpose())]) for p in uniform_planes]

n = 60000

data1 = sio.loadmat('data_mnist_train.mat')
data2 = sio.loadmat('data_mnist_test.mat')
X_train = data1['X_train']
Y_train = data1['Y_train']
X_test = data2['X_test']
Y_test = data2['Y_test']

t = int(time.time())
hash_size = int(rp.s)
input_dim = d*d*1
num_hashtables = 1
tables = []
for i in range(num_hashtables):
    tables.append(dict())

uniform_planes = [np.random.randn(hash_size, input_dim) for _ in range(num_hashtables)]
print(uniform_planes)
np.savez_compressed('hashdb/planes' + str(hash_size) +'.npz', uniform_planes = uniform_planes)

c = 0
output = ""
for j, result in enumerate(X_train):
    #print(len(result))
    hashes = compute_hash(result, uniform_planes)
    for i, hash in enumerate(hashes):
        hash =  hex(int(hash, 2))
        print(hash)
        if hash in tables[i]:
            tables[i][hash].append(c)
        else:
            tables[i][hash] = [c]
    c += 1
np.savez_compressed('hashdb/hashes' + str(hash_size) +'.npz', hash_tables=tables, hash_size=hash_size)
