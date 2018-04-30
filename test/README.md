# Test KD-Tree and LSH on MNIST dataset

## Usage
 * Install imagemagick, python-opencv
 * [Obtain the tiny images dataset](http://horatio.cs.nyu.edu/mit/tiny/data/index.html)
 * Build the hash db. 
`python lsh.py --db /path/to/tiny_images.bin > hashes`

 * Query for an image
`./revImageSearch.sh ../nuke.jpeg`
### 1. generate the original grey scale pictures and labels
      * put the

## Overview of the files
 * lsh.py
```
usage: lsh.py [-h] --db DB [-v]

Build locality sensitve hash database for 80M images

optional arguments:
  -h, --help  show this help message and exit
    --db DB     Path to the tiny images binary
```

 * knn_lsh.py
```
usage: knn_lsh.py [-h] --db DB filename

Find nearest neighbors for image

positional arguments:
  filename

```

 * extract-images.py
```
usage: extract-images.py [-h] --db DB --path PATH [-v] img [img ...]

Extract one or more images from 80M images dataset by index

positional arguments:
  img
```
