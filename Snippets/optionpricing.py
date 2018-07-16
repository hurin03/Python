# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 16:35:42 2017

@author: johan
"""

import numpy
import math
import matplotlib.pyplot as plt


k=100*numpy.cumprod(1+numpy.random.randn(10000,252)*+0.2/math.sqrt(252),1)
print(k.shape)
#for i in k: plt.plot(i)
#plt.show()
#plt.hist(k[:,-1],40)
#plt.show()
v=numpy.mean((k[:,-1]-100)*((k[:,-1]-100)>0))
print(v)
