#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing, threading
import pickle
import numpy as np
from time import gmtime, strftime
from joblib import Parallel, delayed


dbpath = "uploadr/knn/hashdb/"
d = 32
n = 79302017

input_dim = d*d*3

class KnnLSHProcessor:
    def __init__(self, dbCode, imageBinary):
        """ 
            dbCode is the timestamp when the hashdb was created
            imagebinary is the path to the 80M images binary
        """
        self.code = dbCode
        planefile = dbpath+"planes"+str(dbCode)+'.pkl'
        print("Loading the planefile..." + planefile)
        #self.uniform_planes = data["uniform_planes"]
        self.uniform_planes = pickle.load(open(planefile, "rb"))

        hashfile = dbpath+"hashes"+str(dbCode)+".pkl"
        print("Loading the hashfile..." + hashfile)
        data = np.load(hashfile)
        self.hashdb = pickle.load(open(hashfile, "rb"))

        self.imageFile = open(imageBinary)

    def findKNN(self, filename, n, session):

        image = cv2.imread(filename)
        image = cv2.resize(image, (32, 32)) 
        i = np.uint8(np.reshape(image, (d, d, 3), order = 'F'))
        b = list(i[:, :, 0].flatten(order = 'F'))
        g = list(i[:, :, 1].flatten(order = 'F'))
        r = list(i[:, :, 2].flatten(order = 'F'))
        qi = np.int32(np.reshape(r + g + b, (1, d * d * 3), order = 'F'))
        self.qi = qi
        hashes = compute_hash([qi], self.uniform_planes)[0]

        hash = hex(int(hashes[0], 2))
        hashmatch1 = self.hashdb[0][hex(int(hashes[0], 2))]
        #hashmatch2 = self.hashdb[1][hex(int(hashes[1], 2))]
        #hashmatch = list(set(hashmatch1).intersection(hashmatch2))
        hashmatch = hashmatch1
        #hashmatch = range(0,10000)
        session['matches'] = str(len(hashmatch))
        #debugTable = dict()
        #debugTable[hashes[0]] = hashmatch
        #print("Saving debug table")
        #np.savez_compressed(dbpath+'hashes' + str(0) +'.npz', hash_tables=debugTable)

        distances = [None]*len(hashmatch)
        #distances = Parallel(n_jobs=8)(delayed(imageDist)(i, qi, self.imageFile) for i in hashmatch)
        #for i, match in enumerate(hashmatch):
        #    distances[i] = imageDist(i, qi)
        self.imageFile = open(dbpath+self.code+"/"+str(hash)+".bin")
        for i in range(len(hashmatch)):
            sample = self.imageFile.read(d * d * 3)
            sample = np.fromstring(sample, np.uint8)
            sample = np.reshape(sample,(1,d*d*3), order='F')
            distances[i] = distance(sample, self.qi)
        j = 0
        result = {'ind': [], 'dist': []}
        for i in np.argsort(distances):
            j+=1
            result['ind'].append(hashmatch[i])
            result['dist'].append(distances[i])
            if j == n:
                return result

def imageDist(i, b, fh):
    fh.seek(int(i)*d*d*3)
    sample = fh.read(d * d * 3)
    sample = np.fromstring(sample, np.uint8)
    sample = np.reshape(sample,(1,d*d*3), order='F')
    return distance(sample, b)


def distance(a, b):
        return np.linalg.norm(a-b)

def compute_hash(datachunks, planes):
        return [["".join(['1' if i > 0 else '0' for i in np.dot(p, (c-np.full(d*d*3,128)).transpose())]) for p in planes] for c in datachunks]


