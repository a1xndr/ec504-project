#!/usr/bin/env python
# Extract one or more 32x32 images from 80M dataset by index
import cv2, sys, time, os, collections, multiprocessing
import numpy as np
from PIL import Image


d = 32

binpath = "/ec504/80M/tinyimages.bin"
def extract(indices, dest):
    f = open(binpath)
    for n in indices:
        if os.path.isfile(dest+'/'+str(n)+ '.png'):
            continue
        f.seek((int(n))*d*d*3)
        data = f.read(d * d * 3)
        data = np.fromstring(data, np.uint8)
        data = np.reshape(data,(d,d,3), order='F')
        img = Image.fromarray(data, 'RGB')
        img.save(str(dest+'/'+str(n)+ '.png'))

