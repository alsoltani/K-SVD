#coding:latin_1
import numpy as np
import pywt


#---------------------------------------------   WAVELET TRANSFORM CLASS   -----------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

# DictT.dot is a method corresponding to the DWT operator.


class DictT(object):
        def __init__(self, name, level):
            self.name = name
            self.level = level
            self.sizes = []

        def dot(self, mat):
            m = []

            if mat.shape[0] != mat.size:
                for i in xrange(mat.shape[1]):
                    c = pywt.wavedec(mat[:, i], self.name, level=self.level)
                    self.sizes.append(map(len, c))
                    c = np.concatenate(c)
                    m.append(c)
                return np.asarray(m).T
            else:
                c = pywt.wavedec(mat, self.name, level=self.level)
                self.sizes.append(map(len, c))
                return np.concatenate(c)


#-----------------------------------------   INVERSE WAVELET TRANSFORM CLASS   -------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


class Dict(object):
        def __init__(self, name=None, sizes=None):
            self.name = name
            self.sizes = sizes
            assert name, sizes is not None

        def dot(self, m):
            d = []

            if m.shape[0] != m.size:
                for i in xrange(m.shape[1]):
                    sizes_col = self.sizes[i]
                    sizes = np.zeros(len(sizes_col) + 1)
                    sizes[1:] = np.cumsum(sizes_col)
                    c = [m[:, i][sizes[k]:sizes[k + 1]] for k in xrange(0, len(sizes) - 1)]
                    d.append(pywt.waverec(c, self.name))
                return np.asarray(d).T
            else:
                sizes_col = self.sizes[0]
                sizes = np.zeros(len(sizes_col) + 1)
                sizes[1:] = np.cumsum(sizes_col)

                m = [m[sizes[k]:sizes[k + 1]] for k in xrange(0, len(sizes) - 1)]
                return pywt.waverec(m, self.name)
