import os
import numpy as np
from utils import *
from functools import partial
from collections import defaultdict

np.set_printoptions(threshold=np.inf)    # always show the full array


### class for generalized adjacency matrix
class NDSparse:

    ### dictionary that stores the index tuples (key) and the adjacency values (value)
    def __init__(self):
        self.element = {}
        self.tuples = []
        self.values = []

    def addVal(self, tuple, val):
        self.element[tuple] = self.element.get(tuple, 0) + val

    def addArray(self, tuple, arr):
        self.element[tuple] = arr

    def readVal(self, tuple):
        try:
            val = self.element[tuple]
        except KeyError:
            val = 0
        return val

    def readArray(self, tuple):
        try:
            arr = self.element[tuple]
        except KeyError:
            arr = []
        return arr

    ### create an array for index tuples (i, j, ...), and another for corresponding values A(i, j, ...)
    def item2Arrays(self):
        if (len(self.tuples) == 0) and (len(self.values) == 0):
            for tuple, val in self.element.iteritems():
                self.tuples.append(tuple)
                self.values.append(val)
            self.tuples = np.asarray(self.tuples)
            self.values = np.asarray(self.values)
        else:
            print("Error: Tuple and value lists have already been created!")
            return 1

    ### sort the tuples and values arrays in ascending order for tuples
    def sortArrays(self):
        try:
            ind_sorted = np.lexsort((self.tuples[:, 1], self.tuples[:, 0]))
            self.tuples = self.tuples[ind_sorted]
            self.values = self.values[ind_sorted]
        except TypeError:
            print("Error: Convert tuple and value lists to numpy arrays first!")
            return 1


# class Sparse2D:
#     def __init__(self):    ### dictionary that stores the index tuples (key) and the adjacency values (value)
#         self.element = defaultdict(partial(defaultdict, int))

#     def addVal(self, tuple, val):
#         self.element[tuple[0]][tuple[1]] += 1

#     def readVal(self, tuple):
#         val = self.element[tuple[0]][tuple[1]]
#         return val


### first - sixth nearest neighbors
J2 = a_latt / np.sqrt(2)  ### distance
D_J2 = 6   ### degeneracy
D_J3 = 8
D_J4 = 2

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

def findA4(rcut, cell_params, cell_angles_rad, frac_coords):

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

    cells = np.asarray(cells)

    ### set up the generalized adjacency matrix class
    A = NDSparse()

    ### scan subcells inside the supercell
    for I in range(int(Nc[0])):
        for J in range(int(Nc[1])):
            for K in range(int(Nc[2])):

                c_IJK = int(Nc[2] * Nc[1] * I + Nc[2] * J + K)
                cell_IJK = cells[c_IJK]
                trans = np.zeros(3)

                Cts_trans_ijk = NDSparse()
                cells_ijk = NDSparse()

                ### scan subcells that the nearest neighbor atom might reside in
                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ### find the translation vector to map the atoms in the subcell c_ijk to the ijk-th subcell
                            ijk = (i, j, k)

                            for direc in range(3):  ### x, y, z directions
                                if ijk[direc] < 0:
                                    trans[direc] = -1
                                elif ijk[direc] > int(Nc[direc])-1:
                                    trans[direc] = 1
                                else:
                                    trans[direc] = 0

                            ### convert fractional translation vector to Cartesian
                            Cts_trans = np.dot(f2C, trans)
                            err_ind = np.abs(Cts_trans) < rtol
                            Cts_trans[err_ind] = 0.

                            Cts_trans_ijk.addArray(ijk, Cts_trans)

                            ### map the ijk-th subcell to its equivalent subcell (c_ijk) within the supercell
                            c_ijk = int(Nc[2] * Nc[1] * np.mod(i + Nc[0], Nc[0])
                                        + Nc[2] * np.mod(j + Nc[1], Nc[1]) + np.mod(k + Nc[2], Nc[2]))

                            cells_ijk.addArray(ijk, cells[c_ijk])


                # search through each subcell for the three-body kth nearest neighbor clusters
                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ijk = (i, j, k)
    
                            for i2 in range(max(i-1, I-1), min(i+1, I+1) + 1):
                                for j2 in range(max(j-1, J-1), min(j+1, J+1) + 1):
                                    for k2 in range(max(k-1, K-1), min(k+1, K+1) + 1):

                                        ijk2 = (i2, j2, k2)

                                        for i3 in range(max(i2-1, i-1, I-1), min(i2+1, i+1, I+1) + 1):
                                            for j3 in range(max(j2-1, j-1, J-1), min(j2+1, j+1, J+1) + 1):
                                                for k3 in range(max(k2-1, k-1, K-1), min(k2+1, k+1, K+1) + 1):

                                                    ijk3 = (i3, j3, k3)

                                                    ### identify the kth-nearest neighbors with distance = rcut
                                                    for at_1 in cell_IJK:
                                                        for at_2 in cells_ijk.readArray(ijk):
                                                            for at_3 in cells_ijk.readArray(ijk2):
                                                                for at_4 in cells_ijk.readArray(ijk3):    

                                                                    if (np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_2] + Cts_trans_ijk.readArray((ijk)))) - rcut) < rtol  and \
                                                                        np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_3] + Cts_trans_ijk.readArray((ijk2)))) - rcut) < rtol  and \
                                                                        np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_4] + Cts_trans_ijk.readArray((ijk3)))) - rcut) < rtol  and \
                                                                        np.abs(np.linalg.norm((Cts_coords[at_2] + Cts_trans_ijk.readArray((ijk))) - (Cts_coords[at_3] + Cts_trans_ijk.readArray((ijk2)))) - rcut) < rtol  and \
                                                                        np.abs(np.linalg.norm((Cts_coords[at_2] + Cts_trans_ijk.readArray((ijk))) - (Cts_coords[at_4] + Cts_trans_ijk.readArray((ijk3)))) - rcut) < rtol  and \
                                                                        np.abs(np.linalg.norm((Cts_coords[at_3] + Cts_trans_ijk.readArray((ijk2))) - (Cts_coords[at_4] + Cts_trans_ijk.readArray((ijk3)))) - rcut) < rtol):
                                                                        
                                                                        A.addVal((at_1, at_2, at_3, at_4), 1)

    return A


def findA3(rcut, cell_params, cell_angles_rad, frac_coords):

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

    cells = np.asarray(cells)

    ### set up the generalized adjacency matrix class
    A = NDSparse()

    ### scan subcells inside the supercell
    for I in range(int(Nc[0])):
        for J in range(int(Nc[1])):
            for K in range(int(Nc[2])):

                c_IJK = int(Nc[2] * Nc[1] * I + Nc[2] * J + K)
                cell_IJK = cells[c_IJK]
                trans = np.zeros(3)

                Cts_trans_ijk = NDSparse()
                cells_ijk = NDSparse()

                ### scan subcells that the nearest neighbor atom might reside in
                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ### find the translation vector to map the atoms in the subcell c_ijk to the ijk-th subcell
                            ijk = (i, j, k)

                            for direc in range(3):  ### x, y, z directions
                                if ijk[direc] < 0:
                                    trans[direc] = -1
                                elif ijk[direc] > int(Nc[direc])-1:
                                    trans[direc] = 1
                                else:
                                    trans[direc] = 0

                            ### convert fractional translation vector to Cartesian
                            Cts_trans = np.dot(f2C, trans)
                            err_ind = np.abs(Cts_trans) < rtol
                            Cts_trans[err_ind] = 0.

                            Cts_trans_ijk.addArray(ijk, Cts_trans)

                            ### map the ijk-th subcell to its equivalent subcell (c_ijk) within the supercell
                            c_ijk = int(Nc[2] * Nc[1] * np.mod(i + Nc[0], Nc[0])
                                        + Nc[2] * np.mod(j + Nc[1], Nc[1]) + np.mod(k + Nc[2], Nc[2]))

                            cells_ijk.addArray(ijk, cells[c_ijk])


                # search through each subcell for the three-body kth nearest neighbor clusters
                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ijk = (i, j, k)
    
                            for i2 in range(max(i-1, I-1), min(i+1, I+1) + 1):
                                for j2 in range(max(j-1, J-1), min(j+1, J+1) + 1):
                                    for k2 in range(max(k-1, K-1), min(k+1, K+1) + 1):

                                        ijk2 = (i2, j2, k2)

                                        ### identify the kth-nearest neighbors with distance = rcut
                                        for at_1 in cell_IJK:
                                            for at_2 in cells_ijk.readArray(ijk):
                                                for at_3 in cells_ijk.readArray(ijk2):

                                                    if (np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_2] + Cts_trans_ijk.readArray((ijk)))) - rcut) < rtol  and \
                                                        np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_3] + Cts_trans_ijk.readArray((ijk2)))) - rcut) < rtol  and \
                                                        np.abs(np.linalg.norm((Cts_coords[at_2] + Cts_trans_ijk.readArray((ijk))) - (Cts_coords[at_3] + Cts_trans_ijk.readArray((ijk2)))) - rcut) < rtol):
                                                        
                                                        A.addVal((at_1, at_2, at_3), 1)

    return A

### Incorrect implementation of findA3

# def findA3(A, N, rowlen):
#     ### A -- an NDSparse class must have sorted tuples and values arrays
#     A = NDSparse()

#     for i in np.arange(N):
#         for j in A.tuples[i*(rowlen + 1) : (i + 1)*rowlen, 1]:
#             for k in A.tuples[j*(rowlen + 1) : (j + 1)*rowlen, 1]:
#                 if i in A.tuples[k*rowlen : k*(rowlen + 1), 1]:

#                     Aijk = min(A.readVal((i, j)), A.readVal((j, k)), A.readVal((k, i)))

#                     A3.addVal((i, j, k), Aijk)
#                     A3.addVal((i, k, j), Aijk)
#                     A3.addVal((j, i, k), Aijk)
#                     A3.addVal((j, k, i), Aijk)
#                     A3.addVal((k, i, j), Aijk)
#                     A3.addVal((k, j, i), Aijk)

#     return A


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

    cells = np.asarray(cells)

    ### set up the generalized adjacency matrix class
    A = NDSparse()

    ### scan subcells inside the supercell
    for I in range(int(Nc[0])):
        for J in range(int(Nc[1])):
            for K in range(int(Nc[2])):

                c_IJK = int(Nc[2] * Nc[1] * I + Nc[2] * J + K)
                cell_IJK = cells[c_IJK]
                trans = np.zeros(3)

                ### scan subcells that the nearest neighbor atom might reside in
                for i in [I-1, I, I+1]:
                    for j in [J-1, J, J+1]:
                        for k in [K-1, K, K+1]:

                            ### find the translation vector to map the atoms in the subcell c_ijk to the ijk-th subcell
                            ijk = [i, j, k]

                            for direc in range(3):  ### x, y, z directions
                                if ijk[direc] < 0:
                                    trans[direc] = -1
                                elif ijk[direc] > int(Nc[direc])-1:
                                    trans[direc] = 1
                                else:
                                    trans[direc] = 0

                            ### convert fractional translation vector to Cartesian
                            Cts_trans = np.dot(f2C, trans)
                            err_ind = np.abs(Cts_trans) < rtol
                            Cts_trans[err_ind] = 0.

                            ### map the ijk-th subcell to its equivalent subcell (c_ijk) within the supercell
                            c_ijk = int(Nc[2] * Nc[1] * np.mod(i + Nc[0], Nc[0])
                                        + Nc[2] * np.mod(j + Nc[1], Nc[1]) + np.mod(k + Nc[2], Nc[2]))

                            cell_ijk = cells[c_ijk]

                            ### identify the k-nearest neighbors with distance = rcut
                            for at_1 in cell_IJK:
                                for at_2 in cell_ijk:
                                    if np.abs(np.linalg.norm(Cts_coords[at_1] - (Cts_coords[at_2] + Cts_trans)) - rcut) < rtol:
                                        A.addVal((at_1, at_2), 1)

    return A

######################################################################
