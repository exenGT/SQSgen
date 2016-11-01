import osimport numpy as npfrom scipy.sparse import csr_matrixfrom FindAdjacencyMatrix import *from utils import *################## function definitions ##################def findSQS(A, S, D):    ### A should be a sparse matrix    SQS = np.dot(np.transpose(S), A.dot(S)) / (2 * len(S) * D)    return np.abs(SQS)def optimize_SQS(A_J2, A_K2, A_L2, A_M2, D_J2, D_K2, D_L2, D_M2):    A_J2_csr = csr_matrix(A_J2)    A_K2_csr = csr_matrix(A_K2)    A_L2_csr = csr_matrix(A_L2)    A_M2_csr = csr_matrix(A_M2)    ### initialize spin vector S    S_half = np.ones(len(A_J2) / 2)    S = np.concatenate((S_half, np.negative(S_half)))    np.random.shuffle(S)    ### initialization for optimization    Tmax = 200    Tmin = 0.01    Tstep = 0.01    T = Tmax    numStep = 0    tol = 0.01    SQS = findSQS(A_J2_csr, S, D_J2) + findSQS(A_K2_csr, S, D_K2) + findSQS(A_L2_csr, S, D_L2) \          + findSQS(A_M2_csr, S, D_M2)    prevSQS = SQS    S_opt = []    max_len = 5    ### optimization loop    while T > Tmin:        if len(S_opt) == max_len:            break        ### swap two random spin variables in S with opposite signs        S_p = np.where(S == 1.)[0]        S_n = np.where(S == -1.)[0]        a = np.random.choice(S_p)        b = np.random.choice(S_n)        S[b], S[a] = S[a], S[b]        SQS = findSQS(A_J2_csr, S, D_J2) + findSQS(A_K2_csr, S, D_K2) + findSQS(A_L2_csr, S, D_L2) \            + findSQS(A_M2_csr, S, D_M2)        dSQS = SQS - prevSQS        if dSQS <= 0:            prevSQS = SQS            if SQS <= tol:                print(SQS)                S_opt.append(S)                np.random.shuffle(S)        else:            r = np.random.rand()            prob = 1/(1 + np.exp(dSQS/T))            if r <= prob:                prevSQS = SQS            else:                S[b], S[a] = S[a], S[b]        #numStep += 1        #decay = 1./(len(S))        #T *= np.exp(-decay)        T -= Tstep    return S_opt################## call the functions ##################os.chdir("C:\\Users\\pc\\Desktop\\vnl")    # change to your own directoryinfname, cell_params, cell_angles, frac_coords = readcif()print("Cell parameters are ", cell_params)print("Cell angles are ", cell_angles)cell_angles_rad = np.radians(cell_angles)A_J2 = findA(J2, cell_params, cell_angles_rad, frac_coords)A_K2 = findA(K2, cell_params, cell_angles_rad, frac_coords)A_L2 = findA(L2, cell_params, cell_angles_rad, frac_coords)A_M2 = findA(M2, cell_params, cell_angles_rad, frac_coords)#A_N2 = findA(N2, cell_params, cell_angles_rad, frac_coords)#A_O2 = findA(O2, cell_params, cell_angles_rad, frac_coords)S_opt = optimize_SQS(A_J2, A_K2, A_L2, A_M2, D_J2, D_K2, D_L2, D_M2)print(S_opt)for num in range(len(S_opt)):    writecif(infname, num, S_opt[num])# def chosen_S(A, D):####     S_opt = []##     i = 0##     while i < 20:##         S = optimize_SQS(A, D)##         if S != []:##             S_opt.append(S)##             i = i + 1##         else:##             i = i####     return(S_opt)#print('A_N2 = ', chosen_S(A_N2,D_N2))#print('A_O2 = ', chosen_S(A_O2,D_O2))# def final_S():##     final_S = []#     for i in range(20):#         for j in range (20):#             if np.any(chosen_S(A_J2,D_J2)[i] == chosen_S(A_K2,D_K2)[j]):#                 final_S.append(chosen_S(A_J2,D_J2)[i])##     return final_S### final_S = final_S()# print(final_S)#print(len(S_opt_J2))