#!/usr/bin/env python
# References- 
# https://en.wikipedia.org/wiki/K-d_tree
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
                  random.randint(min_val,max_val),
                  random.randint(min_val,max_val)))
    return p

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x1 - x2
    dy = y1 - y2

    return math.sqrt(dx * dx + dy * dy)

def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)

    if d1 < d2:
        return p1
    else:
        return p2

def nns(tree, point, height=0):

    global nearest_nn

    if tree is None:
        return None

    cur_node = tree.loc         
    left = tree.l_child    
    right = tree.r_child

    k = len(point)

    axis = height % k

    next_branch = None
    opposite_branch = None

    if point[axis] < cur_node[axis]:
        next_branch = left
        opposite_branch = right
    else:
        next_branch = right
        opposite_branch = left

    nearest_nn = closer_distance(point,
                           nns(next_branch,point,height+1),
                           cur_node)

    if distance(point, nearest_nn) > abs(point[axis] - cur_node[axis]):
        nearest_nn = closer_distance(point,
                               nns(opposite_branch,point,height+1),
                               nearest_nn)
    return nearest_nn

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
	print(tree)
	print()
	# generate a random query point
	# point = (np.random.normal(random.randint(min_val,max_val), scale=0.5), 
 #        np.random.normal(random.randint(min_val,max_val), scale=0.5),
 #        np.random.normal(random.randint(min_val,max_val), scale=0.5))
	# print('Query Point: ', point)
	# print()
	# initial hyperplace creation
	# hr = [(min_val, max_val), (max_val, min_val)]
	# max_dist = float('inf')
	# find the nearest neighbor
	# nns(tree, point)
    # nns(tree, point)
	# print('Nearest Point in tree: ', nearest_nn)
	# print()

# nearest_nn = None           # nearest neighbor (NN)
# distance_nn = float('inf')  # distance from NN to target

if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print('Total time: ', e-s) # time to run knn search in kd tree
	print()
