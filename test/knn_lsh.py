#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing
#from joblib import Parallel, delayed
import numpy as np
import random
from time import gmtime, strftime
import scipy.io as sio

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))


parser = argparse.ArgumentParser(description="Find nearest neighbors for image")
parser.add_argument('--db', required = True)
parser.add_argument('--code', required = True)
#parser.add_argument('filename', nargs = 1)
rp = parser.parse_args()

dbpath = "hashdb/"

d = 28


def distance(a, b):
        return np.linalg.norm(a-b)

#def compute_hash(datachunks):
#        return [["".join(['1' if i > 0 else '0' for i in np.dot(p, (c-np.full(d*d*1,128)).transpose())]) for p in uniform_planes] for c in datachunks]
def compute_hash(sample, uniform_planes):
    return ["".join(['1' if i > 0 else '0' for i in np.dot(p, (sample-np.full(d*d*1,128)).transpose())]) for p in uniform_planes]

n = 10000
data1 = sio.loadmat('data_mnist_train.mat')
data2 = sio.loadmat('data_mnist_test.mat')
data3 = sio.loadmat('Ypred.mat')
X_train = data1['X_train']
Y_train = data1['Y_train']
X_test = data2['X_test']
Y_test = data2['Y_test']
Y = data3['Ypred']

input_dim = d*d*1

planefile = dbpath+"planes"+str(rp.code)+'.npz'
data = np.load(planefile)
uniform_planes = data["uniform_planes"]
hash_size = len(uniform_planes[0])
print(hash_size)
#this part slow
hashfile = dbpath+"hashes"+str(rp.code)+".npz"
data = np.load(hashfile)
hashdb = data["hash_tables"].item()
print(hashdb.keys())
print('\n', len(hashdb))
#print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# test
Y_pred = []
rangelist = [(1,980),(981,2115),(2116,3147),(3148,4157),(4168,5139),(5140,6031),(6032,6989),(6990,8017),(8018,8991),(8992,10000)]
#idxlist = []
##for i in range(10):
##    for j in range(100):
##        idxlist.append(random.randint(rangelist[i][0]-1, rangelist[i][1]-1))
idxlist = range(n)
X_test2 = []
Y_test2 = []
Y_2 = []
for idx in idxlist:
    X_test2.append(X_test[idx])
for idx in idxlist:
    Y_test2.append(Y_test[idx])
for idx in idxlist:
    Y_2.append(Y[idx])

t = 0
for k in range(n):
    x = X_test2[k]
    qi = np.int32(np.reshape(x, (d, d, 1), order = 'F'))
    hashes = compute_hash(x, uniform_planes)
    #print(hashes)
    for i, h in enumerate(hashes):
        hash = hex(int(h, 2))
        #print(hash)
        if hash in hashdb.keys():
            hashmatch = hashdb[hash]
            #print(len(hashmatch))
            distances = [None]*len(hashmatch)
            closest = 0
            num_cores = 8
            for i, match in enumerate(hashmatch):
                #print(match)
                image = np.uint32(np.reshape(X_train[int(match)], (d, d, 1), order = 'F'))
                distances[i] = distance(image, qi)
            
            knn = [0]*10
            for j in np.argsort(distances)[:10]:
                idx = hashmatch[j]
                pl = Y_train[idx][0]
                knn[pl] += 1
            Y_pred.append(knn.index(max(knn)))
        else:
            print('Use nearest point instead.')
            Y_pred.append(Y_2[k][0])
            t += 1
    print('Label', k, Y_pred[-1])

s = 0
for i in range(n):
    if(Y_pred[i]==Y_test2[i][0]):
        s += 1
print(s)
print('\n', t)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))