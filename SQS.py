import numpy as np
from FindAdjacencyMatrix import *
from utils import *

################## function definitions ##################

def findSQS4(A, S, D, N):
    ### A should be a sparse matrix class object NDSparse

    SQS = 0

    for i in np.arange(len(A.tuples)):
        SQS += A.values[i] * S[A.tuples[i][0]] * S[A.tuples[i][1]] * S[A.tuples[i][2]] * S[A.tuples[i][3]]

    SQS /= (24. * N * D)

    return SQS


def findSQS3(A, S, D, N):
    ### A should be a sparse matrix class object NDSparse

    SQS = 0

    for i in np.arange(len(A.tuples)):
        SQS += A.values[i] * S[A.tuples[i][0]] * S[A.tuples[i][1]] * S[A.tuples[i][2]]

    SQS /= (6. * N * D)

    return SQS


def findSQS(A, S, D, N):
    ### A should be a sparse matrix class object NDSparse

    SQS = 0

    for i in np.arange(len(A.tuples)):
        SQS += A.values[i] * S[A.tuples[i][0]] * S[A.tuples[i][1]]

    SQS /= (2. * N * D)

    return SQS


def changeSQS(A, S, a, b, D, N, rowlen):
    ### calculate the change in correlation function after flippling the spins S[a] and S[b]
    ### A should be a sparse matrix class object NDSparse

    ### tuples and values must be sorted first using A.sortArrays

    ### dSQS = 2(S_b - S_a)[sum_{i!=a, b}(S_i * A_{ai}) - sum_{j!=a, b}(S_j * A_{bj})]
    dSQS = 0

    ### calculate sum_{i!=a, b}(S_i * A_{ai})
    for i in np.arange(a * rowlen, (a + 1) * rowlen):
        if (A.tuples[i][1] != a) and (A.tuples[i][1] != b):
            dSQS += S[A.tuples[i][1]] * A.values[i]
            # print(dSQS)   ### DEBUG

    # print("1. Now the dSQS is", dSQS)

    ### calculate sum_{j!=a, b}(S_j * A_{bj})
    for j in np.arange(b * rowlen, (b + 1) * rowlen):
        if (A.tuples[j][1] != a) and (A.tuples[j][1] != b): 
            dSQS -= S[A.tuples[j][1]] * A.values[j]
            # print(dSQS)   ### DEBUG

    # print("2. Now the dSQS is", dSQS)

    dSQS *= 2 * (S[b] - S[a])
    dSQS /= (2. * N * D)

    return dSQS


def optimize_SQS(A_J2, A_K2, A_L2, A_M2, D_J2, D_K2, D_L2, D_M2, N, tol):

    ### initialization for optimization
    Tmax = 2.0
    Tmin = 0.01
    Tstep = 0.01

    T = Tmax
    numStep = 0
    iters = 0

    ### output the tuple indices and adjacency values to numpy arrays
    A_J2.item2Arrays()
    A_K2.item2Arrays()
    A_L2.item2Arrays()
    A_M2.item2Arrays()

    ### sort the adjacency matrices
    A_J2.sortArrays()
    A_K2.sortArrays()
    A_L2.sortArrays()
    A_M2.sortArrays()

    ### set parameter values for rowlen
    rowlen_J2 = len(A_J2.tuples) / N
    rowlen_K2 = len(A_K2.tuples) / N
    rowlen_L2 = len(A_L2.tuples) / N
    rowlen_M2 = len(A_M2.tuples) / N

    # initialize data
    S_opt = []
    SQS_opt = []
    iter_opt = []

    SQS_vals = []

    max_len = 10
    FOUND = 1

    ### initialize spin vector S
    S_half = np.ones(N / 2)
    S = np.concatenate((S_half, np.negative(S_half)))
    np.random.shuffle(S)


    ### optimization routine
    print("Start optimization ...")
    print("-------------------")

    while (T > Tmin):

        if len(S_opt) == max_len:
            break

        while FOUND == 1:

            SQS_J2 = findSQS(A_J2, S, D_J2, N)
            SQS_K2 = findSQS(A_K2, S, D_K2, N)
            SQS_L2 = findSQS(A_L2, S, D_L2, N)
            SQS_M2 = findSQS(A_M2, S, D_M2, N)
        
            #SQS = findSQS(A_J2, S, D_J2, N)
        
            SQS = np.abs(SQS_J2) + np.abs(SQS_K2) + np.abs(SQS_L2) + np.abs(SQS_M2)

            SQS_vals.append(SQS)

            if SQS <= tol:   ### if an SQS is found
                
                ### if the newly-found array already exists in S_opt
                if any([np.array_equal(S, S_stored) for S_stored in S_opt]):
                    print(numStep)
                    print("Found one, but which already exists.")
                    print(SQS)
                    print("-------------------")
                    #print(S)
                    #print(S_stored)
                else:
                    print(numStep)
                    print("Found new!")
                    print(SQS)
                    print("-------------------")
                    S_opt.append(S)
                    SQS_opt.append(SQS)
                    iter_opt.append(iters)

                S = np.random.permutation(S)

            else:
                print(SQS)
                print("-------------------")
                FOUND = 0

            iters += 1

        prevSQS = SQS


        ### swap two random spin variables in S with opposite signs
        S_p = np.where(S == 1.)[0]
        S_n = np.where(S == -1.)[0]

        for a in S_p:

            if (len(S_opt) == max_len) or (FOUND == 1):
                break

            for b in S_n:

                if (len(S_opt) == max_len) or (FOUND == 1):
                    break

                if S[a] != S[b]:   ### spins are of opposite sign

                    dSQS_J2 = changeSQS(A_J2, S, a, b, D_J2, N, rowlen_J2)
                    dSQS_K2 = changeSQS(A_K2, S, a, b, D_K2, N, rowlen_K2)
                    dSQS_L2 = changeSQS(A_L2, S, a, b, D_L2, N, rowlen_L2)
                    dSQS_M2 = changeSQS(A_M2, S, a, b, D_M2, N, rowlen_M2)

                    SQS = np.abs(SQS_J2 + dSQS_J2) + np.abs(SQS_K2 + dSQS_K2) \
                        + np.abs(SQS_L2 + dSQS_L2) + np.abs(SQS_M2 + dSQS_M2)

                    SQS_vals.append(SQS)

                    dSQS = SQS - prevSQS

                    if dSQS <= 0:   ### target function decreased -- flip spins!

                        prevSQS = SQS
                        
                        S[b], S[a] = S[a], S[b]

                        SQS_J2 += dSQS_J2
                        SQS_K2 += dSQS_K2
                        SQS_L2 += dSQS_L2
                        SQS_M2 += dSQS_M2

                        #print("Flipped -")

                        if SQS <= tol:   ### if an SQS is found

                            ### if the newly-found array already exists in S_opt
                            if any([np.array_equal(S, S_stored) for S_stored in S_opt]):
                                print(numStep)
                                print("Found one, but which already exists.")
                                print(SQS)
                                print("-------------------")
                                #print(S)
                                #print(S_stored)

                            else:
                                print(numStep)
                                print("Found new!")
                                print(SQS)
                                print("-------------------")
                                S_opt.append(S)
                                SQS_opt.append(SQS)
                                iter_opt.append(iters)

                            S = np.random.permutation(S)
                            FOUND = 1

                    ### target function not decreased -- 
                    ### flip spins with "temperature"-dependent probability!
                    else:

                        print(SQS)
                        print("-------------------")

                        r = np.random.rand()
                        prob = 1/(1 + np.exp(dSQS/T))

                        if r <= prob:

                            prevSQS = SQS

                            S[b], S[a] = S[a], S[b]

                            SQS_J2 += dSQS_J2
                            SQS_K2 += dSQS_K2
                            SQS_L2 += dSQS_L2
                            SQS_M2 += dSQS_M2

                            #print("Flipped +")

                        #else:
                            #print("Not flipped +")

                else:
                    #print("Spins are of the same sign. Not flipped.")
                    SQS_vals.append(prevSQS)

                iters += 1

        numStep += 1
        #decay = 1./N
        #T *= np.exp(-decay)
        T -= Tstep


    S_opt = np.asarray(S_opt)
    SQS_opt = np.asarray(SQS_opt)
    iter_opt = np.asarray(iter_opt)
    SQS_vals = np.asarray(SQS_vals)

    print("Optimization finished in %d iterations." % iters)
    print("-------------------")
    print("The correlation function value for the structures found are:")
    np.set_printoptions(precision=4)
    print(SQS_opt)

    print(len(SQS_opt))
    print(len(iter_opt))

    return S_opt, SQS_opt, iter_opt, SQS_vals, iters

######################################################################

