#!/usr/bin/env python
# Reference: https://en.wikipedia.org/wiki/K-d_tree
# Simple implentation of knn of KD Tree

from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2, sys, time, os, argparse, multiprocessing

class Node(namedtuple('Node', 'loc r_child l_child')):

	def __repr__(self):
		return pformat(tuple(self))

def kdtree(pts, height=0):
	""" build kd-tree """
	try:
		k = len(pts[0])
	except IndexError:
		return None

	axis  = height % k

	pts.sort(key=itemgetter(axis))
	median = len(pts)//2

	return Node(
		loc = pts[median],
		r_child = kdtree(pts[median+1:], height+1),
		l_child = kdtree(pts[:median], height+1))
 
def generate_points(n, min_val, max_val):
    """generate a list of random points"""
    p = []
    for i in range(n):
        p.append((random.randint(min_val,max_val),
                  random.randint(min_val,max_val)))
    return p

def nns(tree, point, hp, distance, nearest=None, depth=0):
    """ Find the nearest neighbor for a given point"""
 
    global nearest_nn
    global distance_nn
 
    if tree is None:
        return
 
    k = len(point)
 
    cur_node = tree.loc         
    left = tree.l_child    
    right = tree.r_child 
 
    nearer_kd = further_kd = None
    nearer_hp = further_hp = None
    left_hp = right_hp = None
    axis = depth % k
 
    if axis == 0:
        left_hp = [hp[0], (cur_node[0], hp[1][1])]
        right_hp = [(cur_node[0],hp[0][1]), hp[1]]
 
    if axis == 1:
        left_hp = [(hp[0][0], cur_node[1]), hp[1]]
        right_hp = [hp[0], (hp[1][0], cur_node[1])]
 
    # check which plane the point belongs to
    if point[axis] <= cur_node[axis]:
        nearer_kd = left
        further_kd = right
        nearer_hp = left_hp
        further_hp = right_hp
 
    if point[axis] > cur_node[axis]:
        nearer_kd = right
        further_kd = left
        nearer_hp = right_hp
        further_hp = left_hp
 
    # check if the current node is the closest with the distance formula
    dist = (cur_node[0] - point[0])**2 + (cur_node[1] - point[1])**2
 
    if dist < distance:
        nearest = cur_node
        distance = dist
 
    # go deeper in the tree
    nns(nearer_kd, point, nearer_hp, distance, nearest, depth+1)
 
    # find if there is a closer point in the leaf node
    if distance < distance_nn:
        nearest_nn = nearest
        distance_nn = distance
 
    # a nearer point (px,py)
    px = compute(point[0], further_hp[0][0], further_hp[1][0])
    py = compute(point[1], further_hp[1][1], further_hp[0][1])
 
    # check whether it is closer than the current nearest neighbor
    dist = (px - point[0])**2 + (py - point[1])**2
 
    # explore the further plane if necessary
    if dist < distance_nn:
        nns(further_kd, point, further_hp, distance, nearest, depth+1)

def compute(value, range_min, range_max):
    """ Compute the closest coordinate for the neighboring plane"""
    v = None
    if range_min < value < range_max:
        v = value
    elif value <= range_min:
        v = range_min
    elif value >= range_max:
        v = range_max
    return v
    
def main():
	n = 1000        # number of points
	min_val = 0   # minimal coordinate value
	max_val = 500  # maximal coordinate value
	point_list = generate_points(n, min_val, max_val)
	# construct a K-D tree
	tree = kdtree(point_list)
	# print tree
	# print(tree)
	print()
	# generate a random query point
	point = (np.random.normal(random.randint(min_val,max_val), scale=0.5), np.random.normal(random.randint(min_val,max_val), scale=0.5))
	print('Query Point: ', point)
	print()
	# initial hyperplace creation
	hr = [(min_val, max_val), (max_val, min_val)]
	max_dist = float('inf')
	# find the nearest neighbor
	nns(tree, point, hr, max_dist)
	print('Nearest Point in tree: ', nearest_nn)
	print()

nearest_nn = None           # nearest neighbor (NN)
distance_nn = float('inf')  # distance from NN to target

if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print('Total time: ', e-s) # time to run knn search in kd tree
	print()
