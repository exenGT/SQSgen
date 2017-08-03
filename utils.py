import numpy as np
import matplotlib.pyplot as plt

### some global variables
a_latt = 5.8687


def readcif():

    #infname = input("Open a cif file: " )    ### python3
    infname = raw_input("Open a cif file: " )    ### python2
    try:
        infile = open(infname, 'r')
    except IOError:
        print("File " + infname + " cannot be opened!")
        return 1
    else:
        if not infname.endswith(('.cif')):
            print("Error: " + infname + ": Invalid input file format.")
            return 2

    cell_params = []
    cell_angles = []
    frac_coords = []

    for line in infile:

        if "_cell_length_" in line:
            linearr = line.split(" ")
            cell_params.append(float(linearr[-1].rstrip("\n")))

        if "_cell_angle_" in line:
            linearr = line.split(" ")
            cell_angles.append(float(linearr[-1].rstrip("\n")))

        if "_atom_site_fract_z" in line:
            while True:
                try:
                    line = next(infile)
                    if "Ga" in line:
                        linearr = line.rstrip("\n").split(" ")
                        frac_coords.append([float(i) for i in linearr[-3:]])
                except StopIteration:
                    break
            break

    print("Cell parameters and coordinates successfully read!")

    infile.close()

    cell_params = np.asarray(cell_params)
    cell_angles = np.asarray(cell_angles)
    frac_coords = np.asarray(frac_coords)  # convert to numpy array

    return infname, cell_params, cell_angles, frac_coords


def readcif_full():

    #infname = input("Open a cif file: " )    ### python3
    infname = raw_input("Open a cif file: " )    ### python2
    try:
        infile = open(infname, 'r')
    except IOError:
        print("File " + infname + " cannot be opened!")
        return 1
    else:
        if not infname.endswith(('.cif')):
            print("Error: " + infname + ": Invalid input file format.")
            return 2

    cell_params = []
    cell_angles = []
    atom_species = []
    frac_coords = []

    for line in infile:

        if "_cell_length_" in line:
            linearr = line.split(" ")
            cell_params.append(float(linearr[-1].rstrip("\n")))

        if "_cell_angle_" in line:
            linearr = line.split(" ")
            cell_angles.append(float(linearr[-1].rstrip("\n")))

        if "_atom_site_fract_z" in line:
            while True:
                try:
                    line = next(infile)
                    linearr = line.rstrip("\n").split(" ")
                    atom_species.append(str(linearr[0]))
                    frac_coords.append([float(i) for i in linearr[-3:]])
                except StopIteration:
                    break
            break

    print("Cell parameters and coordinates successfully read!")

    infile.close()

    cell_params = np.asarray(cell_params)
    cell_angles = np.asarray(cell_angles)
    frac_coords = np.asarray(frac_coords)  # convert to numpy array

    return infname, cell_params, cell_angles, atom_species, frac_coords



def readcif_full_2S():

    #infname = input("Open a cif file: " )    ### python3
    infname = raw_input("Open a cif file: " )    ### python2
    try:
        infile = open(infname, 'r')
    except IOError:
        print("File " + infname + " cannot be opened!")
        return 1
    else:
        if not infname.endswith(('.cif')):
            print("Error: " + infname + ": Invalid input file format.")
            return 2

    cell_params = []
    cell_angles = []
    frac_coords = []

    S = []

    for line in infile:

        if "_cell_length_" in line:
            linearr = line.split(" ")
            cell_params.append(float(linearr[-1].rstrip("\n")))

        if "_cell_angle_" in line:
            linearr = line.split(" ")
            cell_angles.append(float(linearr[-1].rstrip("\n")))

        if "_atom_site_fract_z" in line:
            while True:
                try:
                    line = next(infile)
                    linearr = line.rstrip("\n").split(" ")
                    frac_coords.append([float(i) for i in linearr[-3:]])

                    if "Ga" in line:
                        S.append(1)
                    elif "In" in line:
                        S.append(-1)
                    elif "As" in line:
                        S.append(0)

                except StopIteration:
                    break
            break

    print("Cell parameters and coordinates successfully read!")

    infile.close()

    cell_params = np.asarray(cell_params)
    cell_angles = np.asarray(cell_angles)
    frac_coords = np.asarray(frac_coords)  # convert to numpy array

    S = np.asarray(S)  # convert to numpy array

    return infname, cell_params, cell_angles, frac_coords, S


def readcif2S(infname):

    try:
        infile = open(infname, 'r')
    except IOError:
        print("File " + infname + " cannot be opened!")
        return 1
    else:
        if not infname.endswith(('.cif')):
            print("Error: " + infname + ": Invalid input file format.")
            return 2

    S = []

    for line in infile:

        if "_atom_site_fract_z" in line:
            while True:
                try:
                    line = next(infile)
                    if "Ga" in line:
                        S.append(1)
                    elif "In" in line:
                        S.append(-1)

                except StopIteration:
                    break
            break

    #print("Cif file successfully read into spin vector!")

    infile.close()

    S = np.asarray(S)  # convert to numpy array

    return S


def frac2Cart(cell_params, cell_angles_rad, frac_coords):

    vol = np.sqrt(1 - np.sum(np.cos(cell_angles_rad) ** 2) + 2 * np.prod(np.cos(cell_angles_rad)))
    a, b, c = cell_params
    alpha, beta, gamma = cell_angles_rad

    f2C = np.array([[a, b * np.cos(gamma), c * np.cos(alpha)],
                    [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
                    [0, 0, c * vol / np.sin(gamma)]])

    return f2C


def Cart2frac(cell_params, cell_angles_rad, frac_coords):

    vol = np.sqrt(1 - np.sum(np.cos(cell_angles_rad) ** 2) + 2 * np.prod(np.cos(cell_angles_rad)))
    a, b, c = cell_params
    alpha, beta, gamma = cell_angles_rad

    C2f = np.array([[1. / a, -np.cos(gamma)/(a*np.sin(gamma)), (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (a * vol * np.sin(gamma))],
                    [0, 1. / (b * np.sin(gamma)), (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (b * vol * np.sin(gamma))],
                    [0, 0, np.sin(gamma) / (c * vol)]])

    return C2f


def writecif(infname, num, S):

    outfname = infname + str(2*len(S)) + "_" + str(num) + ".cif"

    with open(outfname, 'w') as outfile, open(infname, 'r') as infile:

        count = -1

        for line in infile:
            if "Ga" not in line:
                outfile.write(line)
            else:
                count += 1
                if S[count] == -1.:
                    outfile.write(line.replace("Ga", "In"))
                else:
                    outfile.write(line)

    print("Cif file " + outfname + " written successfully!")


def hist_SQS(SQS_vals):

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 20

    plt.figure(figsize=(8,5))

    plt.hist(SQS_vals, bins=np.arange(0., 1.02, 0.02))
    plt.gca().xaxis.grid(True)
    plt.title("")
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.xlim(0, 1)
    plt.xlabel("target function value")
    plt.ylabel("Frequency")
    
    plt.savefig('histogram_SQS.eps', bbox_inches='tight', dpi=100)
    plt.show()


def traj_SQS(SQS_opt, iter_opt, SQS_vals, iters):

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 20

    plt.figure(figsize=(8,5))

    plt.scatter(np.arange(iters), SQS_vals, s=3, c='blue', alpha=0.7)
    plt.scatter(iter_opt, SQS_opt, s=15, c='red', edgecolors='none')

    plt.title("")
    plt.xlim(-100, iters+100)
    plt.xlabel("iteration")
    plt.ylabel("target function value")

    plt.savefig('trajectory_SQS.eps', bbox_inches='tight', dpi=150)
    plt.show()


def Similarity(S1, S2):
    # uses cosine similarity
    return np.dot(S1, S2)/(np.linalg.norm(S1)*np.linalg.norm(S2))
