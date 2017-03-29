import os
from utils import *
from FindAdjacencyMatrix import *
from SQS import *

np.set_printoptions(threshold=np.inf)    # always show the full array


os.chdir("/Volumes/KINGSTON/SQS_cif")    # change to your own directory

infname, cell_params, cell_angles, frac_coords = readcif()

print("Cell parameters are ", cell_params)
print("Cell angles are ", cell_angles)

cell_angles_rad = np.radians(cell_angles)

# A_J2 = findA(J2, cell_params, cell_angles_rad, frac_coords)
# A_K2 = findA(K2, cell_params, cell_angles_rad, frac_coords)
# A_L2 = findA(L2, cell_params, cell_angles_rad, frac_coords)
# A_M2 = findA(M2, cell_params, cell_angles_rad, frac_coords)

##### print out the contents of A_J2 #####

N = len(frac_coords)

# AJ2 = np.zeros((N, N))

# for i in np.arange(N):
#  	for j in np.arange(N):
#  		AJ2[i, j] = A_J2.readVal((i, j))

# print(AJ2)

# val_total = 0

# for tuple, val in A_J2.element.iteritems():
# 	val_total += val

# print(val_total)

##########################################

# A_J2.item2Arrays()
# A_K2.item2Arrays()
# A_L2.item2Arrays()
# A_M2.item2Arrays()

##### create a spin vector of length N #####

S_half = np.ones(N / 2)
S = np.concatenate((S_half, np.negative(S_half)))
np.random.shuffle(S)

##S = np.array([ 1., -1.,  1.,  1., -1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,
##       -1.,  1.,  1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,
##       -1., -1., -1.,  1.,  1., -1.])

# S = readcif2S()

# print(S)

# SQS_J2 = findSQS(A_J2, S, D_J2, N)
# SQS_K2 = findSQS(A_K2, S, D_K2, N)
# SQS_L2 = findSQS(A_L2, S, D_L2, N)
# SQS_M2 = findSQS(A_M2, S, D_M2, N)

# SQS_J2 = findSQS(A_J2, S, D_J2, N)

# SQS = np.abs(SQS_J2) + np.abs(SQS_K2) + np.abs(SQS_L2) + np.abs(SQS_M2)

# print("SQS = %.4f" % SQS)    ### DEBUG

#A_J2.sortArrays()
# A_K2.sortArrays()
# A_L2.sortArrays()
# A_M2.sortArrays()

# print(A_J2.tuples)
# print(A_J2.values)

# rowlen_J2 = len(A_J2.tuples) / N
# rowlen_K2 = len(A_K2.tuples) / N
# rowlen_L2 = len(A_L2.tuples) / N
# rowlen_M2 = len(A_M2.tuples) / N


# S_p = np.where(S == 1.)[0]
# S_n = np.where(S == -1.)[0]

# a = np.random.choice(S_p)
# b = np.random.choice(S_n)

# a = 1
# b = (N / 2) + 1

# dSQS_J2 = changeSQS(A_J2, S, a, b, D_J2, N, rowlen_J2)
# dSQS_K2 = changeSQS(A_K2, S, a, b, D_K2, N, rowlen_K2)
# dSQS_L2 = changeSQS(A_L2, S, a, b, D_L2, N, rowlen_L2)
# dSQS_M2 = changeSQS(A_M2, S, a, b, D_M2, N, rowlen_M2)

# SQS_J2 += dSQS_J2
# SQS_K2 += dSQS_K2
# SQS_L2 += dSQS_L2
# SQS_M2 += dSQS_M2

# SQS = np.abs(SQS_J2) + np.abs(SQS_K2) + np.abs(SQS_L2) + np.abs(SQS_M2)

# print("SQS = %.4f" % SQS)    ### DEBUG

A_J4 = findA4(J2, cell_params, cell_angles_rad, frac_coords)

val_total = 0

for tuple, val in A_J4.element.iteritems():
	val_total += val

print(val_total / (24. * D_J4 * N))

A_J4.item2Arrays()

##### DEBUG #####
# S[b], S[a] = S[a], S[b]

# SQS_J2 = findSQS(A_J2, S, D_J2, N)
# SQS_K2 = findSQS(A_K2, S, D_K2, N)
# SQS_L2 = findSQS(A_L2, S, D_L2, N)
# SQS_M2 = findSQS(A_M2, S, D_M2, N)

SQS_J4 = findSQS4(A_J4, S, D_J4, N)

# print(SQS_J2)
# print(SQS_K2)
# print(SQS_L2)
# print(SQS_M2)

# SQS2 = np.abs(SQS_J2) + np.abs(SQS_K2) + np.abs(SQS_L2) + np.abs(SQS_M2)

# print("SQS2 = %.4f" % SQS2)

print("SQS_J4 = %.4f" % np.abs(SQS_J4))

##### DEBUG #####


print("Done!")