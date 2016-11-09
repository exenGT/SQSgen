import numpy as np

### some global variables
a_latt = 5.6537

def readcif():
    while True:
        infname = input("Open a cif file: " )
        try:
            infile = open(infname, 'r')
        except IOError:
            print("File " + infname + " cannot be opened!")
            continue
        else:
            if not infname.endswith(('.cif')):
                print('Error: Invalid input file format.')
                continue
            else:
                break

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

    outfname = "SQS--" + str(2*len(S)) + "_" + str(num) + ".cif"

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
