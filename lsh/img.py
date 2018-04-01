#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--db', required = True)
parser.add_argument('-v', action = 'store_true')
parser.add_argument('n', nargs = 1)
rp = parser.parse_args()

verbose = rp.v
f = open(rp.db)
d = 32


# -------------------------------------------------------------------


#f.seek((int(rp.n[0])-1)*d*d*3)
f.seek((int(rp.n[0])-1)*d*d*3)
data = f.read(d * d * 3)
data = np.fromstring(data, np.uint8)
print(data)
data = np.reshape(data,(d,d,3), order='F')
print(data)
img = Image.fromarray(data, 'RGB')
img.save('my'+str(rp.n[0])+'.png')

