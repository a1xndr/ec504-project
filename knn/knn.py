#!/usr/bin/env python


import math
import csv
import argparse
from sys import argv




parser = argparse.ArgumentParser(description="Find nearest neighbors for image")
parser.add_argument('--k', required = True)
parser.add_argument('--db', required = True)
parser.add_argument('filename', nargs = 1)

rp = parser.parse_args()



def distance(trainX, inputimageX):
	y = []
	if (len(trainX) != len(inputimageX)):
		print ("images are not the same size!")
		return None
	eucl = 0
	for i in range(len(trainX[0])):
		eucl += (int(trainX[i],16) - int(inputimageX[i],16))**2
	return euc1



def knn(datafile, inputimageX, k):
	distarray = []

	data = open(datafile).readlines()
	for index, line in enumerate(data.readlines()):
		dist = distance(datafile, inputimageX)
		distarray.append((dist, index))
	sorted(distarray)
	data.close()

	return distarray[0:k-1]



f = open(rp.db)
nearestfiles = []
inputfile = open(rp.filename[0])
for line in inputfile:
	distances = knn(rp.db, line, rp.k)
	for x in distances:
		nearestfiles.append(rp.db[x[1]])
f.close()
inputfile.close()

print (i for i in nearestfiles)