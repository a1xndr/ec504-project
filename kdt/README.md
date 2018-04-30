# KD Tree Implementation
## Usage

 * Install numpy, os, matplotlib.pyplot, sklearn.model_selection, cv2, calendar, time, pickle, PIL, sys, sklearn.decomposition, PIL
 * Important ==> script is written in Python 3 so must have Python 3 to run

## Overview of the files
 * KDT.ipynb
```
usage: open with jupyter notebook

Make sure the directory "cifar-10-batches-py" is in the same directory as this jupyter notebook.

This notebook shows the implementation of NNS of KD-tree with a test image 
from Cifar-10 testImage batch file.
```

 * KDT.py
```
usage: KDT.py k queryImagefile

For example, in terminal just run "python3 KDT.py 10 plane.jpg".

Find k nearest neighbors for queryImagefile.
```

 * pca
```
An output file of all the PCA arrays for the images that implemented into KD Tree.

```

 * cifar-10-batches-py
```
A folder that contains the CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 

There are 50000 training images and 10000 test images. The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. 

They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

Download Link: https://www.cs.toronto.edu/~kriz/cifar.html

```

## References
 * https://en.wikipedia.org/wiki/K-d_tree
 * http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/11_NN_Regularization/11_NN_Regularization.html
 * https://github.com/dhingratul/k-Nearest-Neighbors/blob/master/kNN.py
 * https://github.com/hudara/cifar-10



