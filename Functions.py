#coding:latin_1

import numpy as np
from skimage.util.shape import *
from operator import mul, sub
from math import floor, sqrt, log10
import sys
from scipy.sparse.linalg import svds
from scipy.stats import chi2
from skimage.util import pad
import timeit


#-------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- PSNR TOOLS ---------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def psnr(original_image, approximation_image):
    return 20*log10(np.amax(original_image)) - 10*log10(pow(np.linalg.norm(original_image - approximation_image), 2)
                                                        / approximation_image.size)


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- PATCH CREATION -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def patch_matrix_windows(img, window_shape, step):
    patches = view_as_windows(img, window_shape, step=step)
    cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
    for i in xrange(patches.shape[0]):
        for j in xrange(patches.shape[1]):
            cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
    return cond_patches, patches.shape


def image_reconstruction_windows(mat_shape, patch_mat, patch_sizes, step):
    img_out = np.zeros(mat_shape)
    for l in xrange(patch_mat.shape[1]):
        i, j = divmod(l, patch_sizes[1])
        temp_patch = patch_mat[:, l].reshape((patch_sizes[2], patch_sizes[3]))
        img_out[i*step:(i+1)*step, j*step:(j+1)*step] = temp_patch[:step, :step].astype(int)
    return img_out


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------- MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT -----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def single_channel_omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    res = np.linalg.norm(vect_y)
    atoms_list = []

    while res/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < phi.shape[1]:
        vect_c = phi.T.dot(vect_y - phi.dot(vect_sparse))
        i_0 = np.argmax(np.abs(vect_c))
        atoms_list.append(i_0)
        vect_sparse[i_0] += vect_c[i_0]

        # Orthogonal projection.
        index = np.where(vect_sparse)[0]
        vect_sparse[index] = np.linalg.pinv(phi[:, index]).dot(vect_y)
        res = np.linalg.norm(vect_y - phi.dot(vect_sparse))

    return vect_sparse, atoms_list


def cholesky_omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    atoms_list = []
    vect_alpha = phi.T.dot(vect_y)
    res = vect_y
    l = np.ones((1, 1))
    count = 1

    while np.linalg.norm(res)/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < vect_sparse.shape[0]:

        c = phi.T.dot(res)
        i_0 = np.argmax(np.abs(c))

        if count > 1:
            w = np.linalg.solve(l, phi[:, atoms_list].T.dot(phi[:, i_0]))
            l = np.insert(l, l.shape[1], 0, axis=1)
            l = np.insert(l, l.shape[0], np.append(w.T, sqrt(1 - np.linalg.norm(w))), axis=0)

        atoms_list.append(i_0)
        vect_sparse[atoms_list] = np.linalg.solve(l.dot(l.T), vect_alpha[atoms_list])
        res = vect_y - phi[:, atoms_list].dot(vect_sparse[atoms_list])
        count += 1

    return vect_sparse, atoms_list


def batch_omp(phi, vect_y, sigma):

    #Initial values
    vect_alpha = phi.T.dot(vect_y)
    epsilon = pow(np.linalg.norm(vect_y), 2)
    matrix_g = phi.T.dot(phi)

    atoms_list = []
    l = np.ones((1, 1))
    vect_sparse = np.zeros(phi.shape[1])
    delta = 0
    count = 1

    while np.linalg.norm(epsilon)/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < phi.shape[1]:
        i_0 = np.argmax(np.abs(vect_alpha))

        if count > 1:
            w = np.linalg.solve(l, phi.T[atoms_list].dot(phi[:, i_0]))
            l = np.insert(l, l.shape[0], 0, axis=0)
            l = np.insert(l, l.shape[1], 0, axis=1)
            l[-1, :] = np.append(w.T, sqrt(1 - np.linalg.norm(w)))

        atoms_list.append(i_0)
        vect_sparse[atoms_list] = np.linalg.solve(l.dot(l.T), vect_alpha[atoms_list])
        vect_beta = matrix_g[:, atoms_list].dot(vect_sparse[atoms_list])
        vect_alpha = phi.T.dot(vect_y) - vect_beta
        epsilon += - vect_sparse[atoms_list].T.dot(vect_beta[atoms_list]) + delta
        delta = vect_sparse[atoms_list].T.dot(vect_beta[atoms_list])
        count += 1

    return vect_sparse, atoms_list


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY UPDATING METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def dict_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi
    sparse_temp = matrix_sparse

    if len(indexes) > 0:
        phi_temp[:, k][:] = 0

        matrix_e_k = matrix_y[:, indexes] - phi_temp.dot(sparse_temp[:, indexes])
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)

        phi_temp[:, k] = u[:, 0]
        sparse_temp[k, indexes] = np.asarray(v)[0] * s[0]
    return phi_temp, sparse_temp


def approx_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi

    if len(indexes) > 0:
        phi_temp[:, k] = 0
        vect_g = matrix_sparse[k, indexes].T
        vect_d = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].dot(vect_g)
        vect_d /= np.linalg.norm(vect_d)
        vect_g = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].T.dot(vect_d)
        phi_temp[:, k] = vect_d
        matrix_sparse[k, indexes] = vect_g.T
    return phi_temp, matrix_sparse


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def k_svd(phi, matrix_y, sigma, algorithm, n_iter, approx='yes'):
    phi_temp = phi
    matrix_sparse = np.zeros((phi.T.dot(matrix_y)).shape)
    n_total = []

    print '\nK-SVD, with residual criterion.'
    print '-------------------------------'

    for k in xrange(n_iter):
        print "Stage " + str(k+1) + "/" + str(n_iter) + "..."

        def sparse_coding(f):
            t = f+1
            sys.stdout.write("\r- Sparse coding : Channel %d" % t)
            sys.stdout.flush()
            return algorithm(phi_temp, matrix_y[:, f], sigma)[0]

        sparse_rep = map(sparse_coding, range(matrix_y.shape[1]))
        matrix_sparse = np.array(sparse_rep).T
        count = 1

        updating_range = phi.shape[1]

        for j in xrange(updating_range):
            r = floor(count/float(updating_range)*100)
            sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            sys.stdout.flush()
            if approx == 'yes':
                phi_temp, matrix_sparse = approx_update(phi_temp, matrix_y, matrix_sparse, j)
            else:
                phi_temp, matrix_sparse = dict_update(phi_temp, matrix_y, matrix_sparse, j)
            count += 1
        print '\r- Dictionary updating complete.\n'

    return phi_temp, matrix_sparse, n_total


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ DENOISING METHOD -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def denoising(noisy_image, learning_image, window_shape, window_step, sigma, learning_ratio=0.1, ksvd_iter=1):

    # 1. Form noisy patches.
    padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')
    noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, window_shape, window_step)
    padded_lea_image = pad(learning_image, pad_width=window_shape, mode='symmetric')
    lea_patches, lea_patches_shape = patch_matrix_windows(padded_lea_image, window_shape, window_step)
    print 'Shape of dataset    : ' + str(noisy_patches.shape)

    # 2. Form explanatory dictionary.
    k = int(learning_ratio*lea_patches.shape[1])
    indexes = np.random.random_integers(0, lea_patches.shape[1]-1, k)

    basis = lea_patches[:, indexes]
    basis /= np.sum(basis.T.dot(basis), axis=-1)

    print 'Shape of dictionary : ' + str(basis.shape) + '\n'

    # 3. Compute K-SVD.
    start = timeit.default_timer()
    basis_final, sparse_final, n_total = k_svd(basis, noisy_patches, sigma, single_channel_omp, ksvd_iter)
    stop = timeit.default_timer()
    print "Calculation time : " + str(stop - start) + ' seconds.'

    # 4. Reconstruct the image.
    patches_approx = basis_final.dot(sparse_final)
    padded_denoised_image = image_reconstruction_windows(padded_noisy_image.shape,
                                                         patches_approx, noisy_patches_shape, window_step)
    shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
    denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]
    return denoised_image, stop - start, n_total