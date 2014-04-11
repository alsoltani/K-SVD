#coding:latin_1
from Functions import *
from Classes import *
import numpy as np
import cv2
import timeit

#----------------------------------------------------- CHARGING DATA -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

img = cv2.imread('Degas.jpg', 0)

wave_name = 'haar'
wave_level = None
Phi_t = DictT(level=wave_level, name=wave_name)

BasisT = Phi_t.dot(np.identity(img.shape[0]))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))
Basis = BasisT.T

#------------------------------------------------------ K-SVD METHOD -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

sparsity_constraint = 50

start = timeit.default_timer()
phi_out, sparse_out = k_svd(Basis, img, sparsity_constraint, multi_channel_omp)
stop = timeit.default_timer()

print "Calculation time : " + str(stop - start)

img_out = phi_out.dot(sparse_out).astype(int)

cv2.imwrite('Degas_Out_Sparse_' + str(sparsity_constraint) + '.jpg', img_out)