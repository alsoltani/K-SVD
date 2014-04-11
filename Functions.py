#coding:latin_1
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


#------------------------------------------ APPROXIMATION PURSUIT METHODS ------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

#---------------------------------------- A. MULTI-CHANNEL MATCHING PURSUIT ----------------------------------------#

def multi_channel_mp(phi, matrix_y, sparse_constraint):
    phi_t = phi.T
    j = matrix_y.shape[1]  # Number of channels.
    p = phi_t.dot(matrix_y).shape[0]  # Number of sparse matrix rows.

    matrix_sparse = np.zeros((p, j))

    atoms_list = []

    while len(atoms_list) <= sparse_constraint:

        matrix_c = phi_t.dot(matrix_y - phi.dot(matrix_sparse))
        corr_list = np.zeros(p)
        for i in xrange(p):
            corr_list[i] = np.linalg.norm(matrix_c[i, :])

        i_0 = np.argmax(np.abs(corr_list))

        atoms_list.append(i_0)
        matrix_sparse[i_0, :] += matrix_c[i_0, :]

    return matrix_sparse, atoms_list


#----------------------------------- B. MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT ----------------------------------#

def multi_channel_omp(phi, matrix_y, sparse_constraint):

    phi_t = phi.T
    j = matrix_y.shape[1]
    p = phi_t.dot(matrix_y).shape[0]

    matrix_sparse = np.zeros((p, j))

    atoms_list = []

    while len(atoms_list) <= sparse_constraint:

        matrix_c = phi_t.dot(matrix_y - phi.dot(matrix_sparse))
        corr_list = np.zeros(p)
        for i in xrange(p):
            corr_list[i] = np.linalg.norm(matrix_c[i, :])

        i_0 = np.argmax(np.abs(corr_list))

        atoms_list.append(i_0)
        matrix_sparse[i_0, :] += matrix_c[i_0, :]

        # Orthogonal projection.
        index = np.where(matrix_sparse[:, 0])[0]
        matrix_sparse[index] = np.linalg.pinv(phi.dot(np.identity(p)[:, index])).dot(matrix_y)

    return matrix_sparse, atoms_list


#------------------------------------------- DICTIONARY UPDATING METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def dict_update(phi, matrix_y, matrix_sparse, k):

    col_phi = np.atleast_2d(phi[:, k]).T
    row_sparse = np.atleast_2d(matrix_sparse[k])

    matrix_e_k = matrix_y - phi.dot(matrix_sparse) + col_phi.dot(row_sparse)
    u, s, v = np.linalg.svd(matrix_e_k)

    phi[:, k] = u[:, 0]
    index = np.where(matrix_sparse[k])[0]
    matrix_sparse[k][index] = np.asarray(v)[0] * s[0]

    return phi, matrix_sparse


#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------- A. USING OUR MP, OMP IMPLEMENTATIONS ---------------------------------------#

def k_svd(phi, matrix_y, sparse_constraint, algorithm):

    for k in xrange(phi.shape[1]):
        print "Stage " + str(k+1) + "/" + str(phi.shape[1]+1) + "..."

        matrix_sparse = algorithm(phi, matrix_y, sparse_constraint)[0]
        phi, matrix_sparse = dict_update(phi, matrix_y, matrix_sparse, k)

    return phi, matrix_sparse


#------------------------------------------------ B. USING SKLEARN -------------------------------------------------#

def k_svd_sklearn(phi, matrix_y, sparse_constraint):

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparse_constraint)

    for k in xrange(phi.shape[1]):
        print "Stage " + str(k+1) + "/" + str(phi.shape[1]+1) + "..."

        print '*** Approximation Pursuit.'
        omp.fit(phi, matrix_y)
        matrix_sparse = omp.coef_.T

        print '*** Dictionary Update.'
        phi, matrix_sparse = dict_update(phi, matrix_y, matrix_sparse, k)

    return phi, matrix_sparse
