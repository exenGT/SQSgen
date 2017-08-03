import os
import numpy as np
from FindAdjacencyMatrix import *
from SQS import *
from utils import *

################## call the functions ##################

os.chdir("/Users/jw598/Downloads/SQS_cif")    # change to your own directory

infname, cell_params, cell_angles, frac_coords = readcif()

print("Cell parameters are ", cell_params)
print("Cell angles are ", cell_angles)

cell_angles_rad = np.radians(cell_angles)

A_J2 = findA(J2, cell_params, cell_angles_rad, frac_coords)
A_K2 = findA(K2, cell_params, cell_angles_rad, frac_coords)
A_L2 = findA(L2, cell_params, cell_angles_rad, frac_coords)
A_M2 = findA(M2, cell_params, cell_angles_rad, frac_coords)
#A_N2 = findA(N2, cell_params, cell_angles_rad, frac_coords)
#A_O2 = findA(O2, cell_params, cell_angles_rad, frac_coords)

N = len(frac_coords)

tol = 0.1
S_opt, SQS_opt, iter_opt, SQS_vals, iters = optimize_SQS(A_J2, A_K2, A_L2, A_M2, D_J2, D_K2, D_L2, D_M2, N, tol)

#S_opt = MC_SQS(A_J2, A_K2, A_L2, A_M2, D_J2, D_K2, D_L2, D_M2)

hist_SQS(SQS_vals)

traj_SQS(SQS_opt, iter_opt, SQS_vals, iters)

for num in range(len(S_opt)):
    writecif(infname, num, S_opt[num])