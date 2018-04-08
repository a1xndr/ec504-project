#!/usr/bin/env python
# References- 
# https://en.wikipedia.org/wiki/K-d_tree
# simple 2d nns implemented
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
 
def nns(tree, q_point, plane, distance, nearest=None, height=0):
    """ Find the nearest neighbor for the query point"""
 
    global nearest_nn
    global distance_nn
 
    if tree is None:
        return
 
    k = len(q_point)
 
    cur_node = tree.loc        
    left_branch = tree.l_child 
    right_branch = tree.r_child 
 
    nearer_kd = further_kd = nearer_hr = further_hr = left_hr = right_hr  = None
 
    # Select axis based on depth so that axis cycles through all valid values
    axis = height % k
 
    if axis == 0:
        left_hr = [plane[0], (cur_node[0], plane[1][1])]
        right_hr = [(cur_node[0],plane[0][1]), plane[1]]
 
    if axis == 1:
        left_hr = [(plane[0][0], cur_node[1]), plane[1]]
        right_hr = [plane[0], (plane[1][0], cur_node[1])]
 
    if q_point[axis] <= cur_node[axis]:
        nearer_kd = left_branch
        further_kd = right_branch
        nearer_hr = left_hr
        further_hr = right_hr
 
    if q_point[axis] > cur_node[axis]:
        nearer_kd = right_branch
        further_kd = left_branch
        nearer_hr = right_hr
        further_hr = left_hr
 
    dist = (cur_node[0] - q_point[0])**2 + (cur_node[1] - q_point[1])**2
 
    if dist < distance:
        nearest = cur_node
        distance = dist
 
    # go deeper in the tree
    nns(nearer_kd, q_point, nearer_hr, distance, nearest, height+1)
 
    if distance < distance_nn:
        nearest_nn = nearest
        distance_nn = distance
    px = closest(q_point[0], further_hr[0][0], further_hr[1][0])
    py = closest(q_point[1], further_hr[1][1], further_hr[0][1])
 
    dist = (px - q_point[0])**2 + (py - q_point[1])**2
    if dist < distance_nn:
        nns(further_kd, q_point, further_hr, distance, nearest, height+1)

def closest(v, _min, _max):
    """ Compute the closest coordinate for the neighboring plane"""
    x = None
    if _min < v < _max:
        x = v
    elif v <= _min:
        x = _min
    elif v >= _max:
        x = _max
    return x
    
def main():
	n = 1000        # number of points
	min_val = 0   # minimal coordinate value
	max_val = 500  # maximal coordinate value
	point_list = generate_points(n, min_val, max_val)
	# construct a K-D tree
	tree = kdtree(point_list)
	# print tree
	print(tree)
	print()
	# generate a random query point
	point = (np.random.normal(random.randint(min_val,max_val), scale=0.5), 
        np.random.normal(random.randint(min_val,max_val), scale=0.5))
	print('Query Point: ', point)
	print()
	# initial hyperplane creation
	plane = [(min_val, max_val), (max_val, min_val)]
	max_dist = float('inf')
	# find the nearest neighbor
	nns(tree, point, plane, max_dist)
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
