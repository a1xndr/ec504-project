#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename', nargs = 1)
rp = parser.parse_args()

d = 32

i = np.uint8(np.reshape(cv2.imread(rp.filename[0]), (d, d, 3), order = 'F'))
b = list(i[:, :, 0].flatten(order = 'F'))
g = list(i[:, :, 1].flatten(order = 'F'))
r = list(i[:, :, 2].flatten(order = 'F'))
qi = np.int32(np.reshape(r + g + b, (1, d * d * 3), order = 'F'))

n = 79302017

input_dim = d*d*3
num_hashtables = 1

data = np.load('file.npz')
uniform_planes = data["uniform_planes"]
s = ["".join(['1' if i > 0 else '0' for i in np.dot(uniform_planes[0], c.transpose())]) for c in [qi]]

print(s[0])

