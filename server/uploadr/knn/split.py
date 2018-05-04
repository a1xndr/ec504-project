#!/usr/bin/env python

import cv2, sys, time, os, collections, argparse, multiprocessing, time
import pickle
import numpy as np

parser = argparse.ArgumentParser(description="Build locality sensitve hash database for 80M images")
parser.add_argument('--db', required = True, help='Path to the tiny images binary')
rp = parser.parse_args()
f = open(rp.db)
d = 32


start = time.time()

planefile = "hashdb/planes11_1524961711.pkl"
# -------------------------------------------------------------------

def read_stream():
        l = []
        while True:
                data = f.read(d * d * 3)
                if not data or len(data) != d * d * 3:
                        break
                l.append(data)
                if len(l) >= 400:
                        yield l
                        l = []
        yield l

def compute_hash(datachunks):
        result = []
        for c in datachunks:
            result.append(["".join(['1' if i > 0 else '0' for i in np.dot(p, np.fromstring(c, np.uint8)-np.full(d*d*3,128))]) for p in uniform_planes])
        return result

def process(stream):
        q = collections.deque()
        pool = multiprocessing.Pool(processes = 16)
        c = 0
        ls = collections.deque()
        for i in stream:
                for n in i:
                    ls.append(n)
                if len(q) >= 20:
                    r = q.popleft().get()
                    for k in r:
                        hash = hex(int(k[0], 2))
                        if hash not in fh:
                            fh[hash] = open("hashdb/11_1524961711-2/"+str(hash)+".bin","a+")
                    fh[hash].write(ls.popleft())
                    fh[hash].flush()
                    c+= 1
                    yield r
                q.append(pool.apply_async(compute_hash, (i,)))
        for j in enumerate(q):
                r = j.get()
                for k in r:
                    hash = hex(int(k[0], 2))
                    if hash not in fh:
                        fh[hash] = open("hashdb/11_1524961711-2/"+str(hash)+".bin","a+")
                    fh[hash].write(ls.popleft().get())
                c+= 1
                yield r


n = 79302017

t = int(time.time())
input_dim = d*d*3

uniform_planes = [pickle.load(open(planefile, "rb"))[0]]
print uniform_planes

fh = dict()
c = 0
output = ""
for i, j in enumerate(process(read_stream())):
    continue;
