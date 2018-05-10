#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing, threading
import pickle
import numpy as np
from time import gmtime, strftime
from joblib import Parallel, delayed


dbpath = "hashdb/"
d = 32
n = 79302017
code = "11_1524961711"
hashfile = dbpath + "hashes"+code+".pkl"
input_dim = d*d*3
imageBinary = "/ec504/80M/tinyimages.bin"
hashdb = pickle.load(open(hashfile, "rb"))
imageFile = open(imageBinary)
#os.makedirs(dbpath+"/"+code)
key = "0x6b7"
with open(dbpath+"/"+str(code)+"/"+str(key)+".bin","a+") as f:
    data = ""
    for i in hashdb[0][key]:
        imageFile.seek(d*d*3*i)
        data += imageFile.read(d * d * 3)
    f.write(data)
#for key in hashdb[0]:
#   if key != 0x475:
 #      print("Skipping " + str(key))
 #      continue
  # if os.path.isfile(dbpath+"/"+str(code)+"/"+str(key)+".bin"):
  #     print("Skipping " + str(key))
   #    continue;
   #with open(dbpath+"/"+str(code)+"/"+str(key)+".bin","a+") as f:
    #   data = ""
    #   for i in hashdb[0][key]:
     #      imageFile.seek(d*d*3*i)
     #      data += imageFile.read(d * d * 3)
      # f.write(data)
      #
