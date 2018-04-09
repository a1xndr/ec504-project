#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing
import numpy as np

parser = argparse.ArgumentParser(description="Find nearest neighbors for image")
parser.add_argument('--db', required = True)
parser.add_argument('filename', nargs = 1)
rp = parser.parse_args()

d = 32


def distance(a, b):
        return np.linalg.norm(a-b)

def compute_hash(datachunks):
        return ["".join(['1' if i > 0 else '0' for i in np.dot(uniform_planes[0], (c-np.full(d*d*3,128)).transpose())]) for c in datachunks]

n = 79302017

hash_size = 11
input_dim = d*d*3
num_hashtables = 2
data = np.load('file.npz')
uniform_planes = data["uniform_planes"]


i = np.uint8(np.reshape(cv2.imread(rp.filename[0]), (d, d, 3), order = 'F'))
b = list(i[:, :, 0].flatten(order = 'F'))
g = list(i[:, :, 1].flatten(order = 'F'))
r = list(i[:, :, 2].flatten(order = 'F'))
qi = np.int32(np.reshape(r + g + b, (1, d * d * 3), order = 'F'))
hash = compute_hash([qi])
hash = hash[0]

#this part slow
fname = "hashes"
hashmatch = []
with open(fname) as f:
    line = f.readline()
    while line:
        line = line.split(" ")
        if line[0] == hash:
            hashmatch.append(line[-1][:-1])
            #print hashmatch[-1]
        line = f.readline()
f.close()


f = open(rp.db)
distances = [None]*len(hashmatch)
closest = 0
for i, match in enumerate(hashmatch):
    f.seek((int(match)-1)*d*d*3)
    image = f.read(d * d * 3)
    image = np.fromstring(image, np.uint8)
    #print(image)
    image = np.reshape(image,(1,d*d*3), order='F')
    distances[i] = distance(image, qi)

for i in np.argsort(distances):
    print(str(hashmatch[i]) +" : " + str(distances[i]))
f.close()
