#coding:utf-8  
import numpy as np  
import struct  
# import matplotlib.pyplot as plt  
# from scipy.misc import imsave  
   
#filename = 'dat//train-images.idx3-ubyte'  
filename =  'train-labels.idx1-ubyte'  
binfile = open(filename , 'rb')  
buf = binfile.read()  
  
f1 = open('label.txt', 'w')  
  
index = 0  
#'>IIII'使用大端法读取两个unsigned int32  
magic, numLabels = struct.unpack_from('>II' , buf , index)  
index += struct.calcsize('>II')  
  
# 输出大端数  
print (magic)  
print (numLabels)  
  
# for i in range(numLabels):  
for i in range(numLabels):  
    numtemp = struct.unpack_from('1B' ,buf, index)  
    # numtemp 为tuple类型，读取其数值  
    num = numtemp[0]  
    # 存入label.txt文件中  
    f1.write(str(num))  
      
    f1.write('\n')  
    index += struct.calcsize('1B')  
    print (num)  
      
  
   
# fig = plt.figure()  
# plotwindow = fig.add_subplot(111)  
# plt.imshow(im , cmap='gray')  
# plt.show() 