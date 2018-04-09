#!/usr/bin/env python


import math

def knn(trainX, inputimageX):
	y = []
	for j in range(len(tranX)):
		eucl = 0
		dist = []
		for i in range(len(trainX[0])):
			eucl = eucl + (trainX[i] - inputimageX[i])**2
		dist.append([math.sqrt(eucl), j])
	return dist

def dataparse(filenameX, inputimageX):
	trainX = []
	with open(filenameX) as xtrain:
		for line in xtrain:
			newline = line.split()
			num = ""
			for i in newline:
				num += i
			trainX.append(num)
	xtrain.close()

	with open(inputimageX) as xtest:
		testX = []
		for line in xtest:
			newline = line.split()
			num = ""
			for i in newline:
				num += i
			testX.append(num)
	xtest.close()

	return trainX, testX



if __name__ == '__main__':
	filenameX = "data_mnist_train.txt"
	inputimageX = "data_mnist_test.txt"

	trainX, testX = dataparse(filenameX, inputimageX)





