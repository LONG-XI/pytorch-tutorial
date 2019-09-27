#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:49:02 2019

@author: xilong
"""

import torch
import numpy as np

# 2 rows 3 colums
np_data = np.arange(6).reshape((2,3))  
torch_data = torch.from_numpy(np_data) # numpy to torch
tensor2array = torch_data.numpy() # torch to numpy


print('numpy', np_data, 'torch', torch_data)
print('tensor2array', tensor2array)


# http://pytorch.org/docs/torch.html#math-operations
# abs
data =[-1,-2,1,2]
tensor = torch.FloatTensor(data)  # 32bit

print('\nabs', 
      '\nnumpy:', np.abs(data),
      '\ntorch:', torch.abs(tensor))

# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)
## incorrect method
#data = np.array(data)
#print(
#    '\nmatrix multiplication (dot)',
#    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
#    '\ntorch: ', tensor.dot(tensor)     # this will convert tensor to [1,2,3,4], you'll get 30.0
#)