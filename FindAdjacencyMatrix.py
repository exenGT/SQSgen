import os
import numpy as np
from utils import *

np.set_printoptions(threshold=np.inf)    # always show the full array

### some global variables
J2 = a_latt / np.sqrt(2)
D_J2 = 6

K2 = a_latt
D_K2 = 3

L2 = np.sqrt(J2 ** 2 + K2 ** 2)
D_L2 = 12

M2 = a_latt * np.sqrt(2)
D_M2 = 6

N2 = np.sqrt((a_latt/2.) ** 2 + (3.*a_latt/2.) ** 2)
D_N2 = 12

O2 = a_latt * np.sqrt(3)
D_O2 = 4

################# functions #################

### sort atoms in the ascending order of their subcell indices
# def quicksort(arr):
#     less = []
#     eq = []
#     greater = []
#
#     if len(arr) > 1:
#         pivot = arr[0]
#         for x in arr:
#             if x < pivot:
#                 less.append(x)
#             elif x == pivot:
#                 eq.append(x)
#             else:
#                 greater.append(x)
#         return quicksort(less) + eq + quicksort(greater)
#     else:
#         return arr

def findA(rcut, cell_params, cell_angles_rad, frac_coords):

    ### set tolerance for interatomic distance
    rtol = 1e-3

    ### eliminate all small errors in frac_coords
    err_ind = np.abs(frac_coords) < rtol
    frac_coords[err_ind] = 0.

    ### convert all fractional coords to Cartesian coords
    f2C = frac2Cart(cell_params, cell_angles_rad, frac_coords)
    Cts_coords = np.transpose(np.dot(f2C, np.transpose(frac_coords)))

    ### eliminate all small errors in Cts_coords
    err_ind = np.abs(Cts_coords) < rtol
    Cts_coords[err_ind] = 0.

    ### determine the subcell dimensions and number of subcells inside the supercell
    Lc = [rcut / np.amin(np.sin(np.delete(cell_angles_rad, i))) for i in range(3)]
    Nc = np.floor(cell_params / Lc)

    ### assign each atom to its respective subcell
    cells = [[] for cell in range(int(np.prod(Nc)))]

    N = len(Cts_coords)
    for at_ind in range(N):    # atom index
        r = frac_coords[at_ind]
        c_r = int(Nc[2] * Nc[1] * np.floor(r[0]*Nc[0]) + Nc[2] * np.floor(r[1]*Nc[1]) + np.floor(r[2]*Nc[2]))
        cells[c_r].append(at_ind)

    np.asarray(cells)

    ### set up the generalized adjacency matrix
    A = np.zeros((N, N))

    ### scan subcells inside the supercell
    for I in range(int(Nc[0])):
        for J in range(int(Nc[1])):
            for K in range(int(Nc[2])):

                c_IJK = int(Nc[2] * Nc[1] * I + Nc[2] * J + K)
                cell_IJK = cells[c_IJK]
                trans = np.zeros(3)

                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ### map the ijk-th subcell to its equivalent subcell (c_ijk) in the supercell
                            c_ijk = int(Nc[2] * Nc[1] * np.mod(i + Nc[0], Nc[0])
                                        + Nc[2] * np.mod(j + Nc[1], Nc[1]) + np.mod(k + Nc[2], Nc[2]))

                            cell_ijk = cells[c_ijk]

                            ### find the translation vector to map the atoms in the subcell c_ijk to the ijk-th subcell
                            ijk = [i, j, k]

                            for direc in range(3):
                                if ijk[direc] < 0:
                                    trans[direc] = -1
                                elif ijk[direc] > int(Nc[direc])-1:
                                    trans[direc] = 1
                                else:
                                    trans[direc] = 0

                            Cts_trans = np.dot(f2C, trans)
                            err_ind = np.abs(Cts_trans) < rtol
                            Cts_trans[err_ind] = 0.

                            ### identify the k-nearest neighbors with distance = rcut
                            for at_1 in cell_IJK:
                                for at_2 in cell_ijk:
                                    if np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_2] + Cts_trans)) - rcut) < rtol:
                                        A[at_1, at_2] += 1

    return A

################## DEBUG ##################
# def testSQS(A, N, D):
#     S = np.ones(N)
#     return np.dot(np.transpose(S), np.dot(A, S)) / (2*N*D)
#
# SQS_J2 = testSQS(A_J2, len(frac_coords), D_J2)
# print(SQS_J2)
###########################################

#A_K2 = findA(K2, cell_params, cell_angles, frac_coords)
#print(A_K2)
#A_L2 = findA(L2, cell_params, cell_angles, frac_coords)
#print(A_L2)
#A_M2 = findA(M2, cell_params, cell_angles, frac_coords)
#print(A_M2)
#A_N2 = findA(N2, cell_params, cell_angles, frac_coords)
#print(A_N2)

###########################################