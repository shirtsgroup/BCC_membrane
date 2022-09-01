#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import mdtraj as md
from scipy.optimize import fsolve
from gromacs.formats import XVG

######################################################################################
###################################### INPUTS ########################################
######################################################################################

cwd = './T343/' # current working directory where all files are located and should be saved
xtc = cwd + 'prod01.xtc' # trajectory file
gro = cwd + 'prod1.gro' # topology file

start = 0 # starting frame (for glycerol solvated, 2001 total frames that last 500 ns)
step = 1 # frame step (for glycerol solvated, frames saved every 0.25 ns)

ts = True # time series of distance to minimal surface

ndens = False
n_bins = 200
plot = False
test_bilayer = False
test_solver = False

######################################################################################
####################### Functions for performing analysis ############################
######################################################################################

def plot_implicit(fn, bbox=(-2.5,2.5), a=1):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 100) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=a, colors='slategray')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', alpha=a, colors='slategray')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', alpha=a, colors='slategray')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    return fig, ax


# Define the triply periodic minimal surface functions
def Gyroid(X,period):
    
    N = 2*np.pi/period
    
    a = np.sin(N*X[0]) * np.cos(N*X[1])
    b = np.sin(N*X[1]) * np.cos(N*X[2])
    c = np.sin(N*X[2]) * np.cos(N*X[0])
    
    return a + b + c

def Gyroid_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a =  N*np.cos(N*x)*np.cos(N*y) - N*np.sin(N*x)*np.sin(N*z)
    b = -N*np.sin(N*y)*np.sin(N*x) + N*np.cos(N*y)*np.cos(N*z)
    c = -N*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*z)*np.cos(N*x)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def magnitude_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a =  N*np.cos(N*x)*np.cos(N*y) - N*np.sin(N*x)*np.sin(N*z)
    b = -N*np.sin(N*y)*np.sin(N*x) + N*np.cos(N*y)*np.cos(N*z)
    c = -N*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*z)*np.cos(N*x)
    
    return np.linalg.norm(np.array([a,b,c]))

def Gyroid_solve(solution, params):

    '''
    Function to give to fsolve to solve system of equations

    Input params is dictionary with variables to solve for (alpha, x, y, z) and input constants (period, x0, y0, z0)
    '''

    # input constants
    period = params['period'] # period for gyroid
    x0, y0, z0 = params['point'] # point off the surface to find the distance to surface

    # variables to solve for
    alpha, x, y, z = solution
    
    N = 2*np.pi / period
    abs_grad = magnitude_grad([x,y,z], period)

    eq1 = x - x0 - N*alpha * (np.cos(N*x)*np.cos(N*y) - np.sin(N*x)*np.sin(N*z)) / abs_grad
    eq2 = y - y0 - N*alpha * (-np.sin(N*y)*np.sin(N*x) + np.cos(N*y)*np.cos(N*z)) / abs_grad
    eq3 = z - z0 - N*alpha * (-np.sin(N*y)*np.sin(N*z) + np.cos(N*z)*np.cos(N*x)) / abs_grad
    eq4 = Gyroid([x,y,z], period)

    return (eq1, eq2, eq3, eq4)


def plane_solve(solution, params):

    '''
    Function to give to fsolve to solve system of equations

    Input params is dictionary with variables to solve for (alpha, x, y, z) and input constants (period, x0, y0, z0)

    For a bilayer, the ideal surface is F(x,y,z) = z - (center of solvent) = 0
    So for atomistic bilayer simulations, we approximate the bilayer to be centered within the box, so (center of solvent) = 0
    '''

    # input constants
    period = params['period'] # z dimension of box
    x0, y0, z0 = params['point'] # point off the surface to find the distance to surface

    # variables to solve for
    alpha, x, y, z = solution
    
    eq1 = x - x0 - alpha * 0
    eq2 = y - y0 - alpha * 0
    eq3 = z - z0 - alpha * 1
    eq4 = z #- period

    return (eq1, eq2, eq3, eq4)


def calculate_deviation(traj, selection, label, n_atoms_per_monomer=None, c='black', eq_solve=Gyroid_solve, ts=True, ndens=True, filename='output.xvg', n_bins=50, plot=False, verbose=False):

    # Get information from the trajectory
    top = traj.top
    coords = traj.xyz
    n_frames = traj.n_frames
    box = traj.unitcell_lengths

    # Define a topology with only the selection
    sel_atoms = top.select(selection)
    sel = top.subset(sel_atoms)

    # Allocate some arrays
    if n_atoms_per_monomer is None: # this is an input for the number of atoms in the selection that would belong to a single monomer
        com = np.zeros((n_frames, sel.n_residues, 3))
        deviations = np.zeros((n_frames, sel.n_residues))
        intersect_points = np.zeros((n_frames, sel.n_residues, 3))

    else:
        tot_monomers = int(len(sel_atoms) / n_atoms_per_monomer)
        com = np.zeros((n_frames, tot_monomers, 3))
        deviations = np.zeros((n_frames, tot_monomers))
        intersect_points = np.zeros((n_frames, tot_monomers, 3))

    histograms = np.zeros((n_frames, n_bins-1))
    bin_edges = np.zeros((n_frames, n_bins))


    max_dev = -100
    min_dev = 100
    for frame in range(n_frames):

        n_monomer = 0
        first_atom = None
        if sel.n_residues == 1:
            # Calculate the center of mass based on atom index (should only be for crosslinked monomer selections)
            for atom in sel.residue(0).atoms:
                full_idx = sel_atoms[atom.index]
                atom_coords = coords[frame, full_idx, :]

                if not first_atom:            # save first atom in selection group
                    first_atom = atom.name
                    first_atom_coords = atom_coords
                    mass = 0
                elif atom.name == first_atom: # finalize previous COM and increment to next monomer whenever we encounter the first atom again
                    com[frame,n_monomer,:] = com[frame,n_monomer,:] / mass

                    n_monomer += 1
                    mass = 0
                    first_atom_coords = atom_coords

                r = first_atom_coords - atom_coords
                for d in range(3):
                    if r[d] < -box[frame,d] / 2: # if current atom is on other side of box from first_atom and first_atom < atom_coords
                        atom_coords[d] -= box[frame,d]
                    elif r[d] > box[frame,d] / 2: # if current atom is on other side from first_atom and first_atom > atom_coords
                        atom_coords[d] += box[frame,d]

                com[frame,n_monomer,:] += atom.element.mass * atom_coords
                mass += atom.element.mass

        else:
            # Calculate the center of mass for each residue in the selection
            for n_res, res in enumerate(sel.residues):
                mass = 0
                res_full_top = top.residue(res.resSeq - 1) # get the residue info from original topology to get correct coordinates
                first_atom = np.array([0,0,0])
                for atom in res_full_top.atoms:
                    if atom.index in sel_atoms:
                        atom_coords = coords[frame, atom.index, :]
                        if first_atom[0] == 0 and first_atom[1] == 0 and first_atom[2] == 0: # save the position of first atom
                            first_atom = atom_coords
                        
                        r = first_atom - atom_coords
                        for d in range(3):
                            if r[d] < -box[frame,d] / 2: # if current atom is on other side of box from first_atom and first_atom < atom_coords
                                atom_coords[d] -= box[frame,d]
                            elif r[d] > box[frame,d] / 2: # if current atom is on other side from first_atom and first_atom > atom_coords
                                atom_coords[d] += box[frame,d]                        
                            
                        com[frame,n_res,:] += atom.element.mass * atom_coords
                        mass += atom.element.mass
                
                com[frame,n_res,:] = com[frame,n_res,:] / mass


        # Calculate the deviation distance for each center of mass
        period = box[frame, 2]
        for n_res, point in enumerate(com[frame,:,:]):
            params = {
                'period' : period,
                'point'  : point
            }
            guess_dist = 1 # guess the distance from the surface
            guess_point = point # guess the point on surface normal to center of mass point
            guess = (guess_dist, guess_point[0], guess_point[1], guess_point[2])

            sol = fsolve(eq_solve, guess, args=params)
            deviations[frame, n_res] = sol[0]
            intersect_points[frame, n_res,:] = np.array([sol[1], sol[2], sol[3]])
            if verbose and n_res % 200 == 0:
                print('Calculated the deviation for %d out of %d molecules on frame %d' %(n_res+1, sel.n_residues, frame+1) )

            # if max(deviations[frame,:]) > max_dev:
            #     max_dev = max(deviations[frame,:])
            # if min(deviations[frame,:]) < min_dev:
            #     min_dev = min(deviations[frame,:])
        
        if frame % 20 == 0:
            print('Completed deviation calculations for frame %d out of %d!' %(frame+1, n_frames))

    if ts: # Save the average deviation for each frame over time
        deviations = abs(deviations) # look at absolute value or else the average will be close to 0
        
        start = traj.time[0]
        stop = traj.time[-1]
        step = (stop - start) / (n_frames-1)
        time = np.arange(start, stop+step, step)

        data = np.array([time, np.mean(abs(deviations), axis=1)])
        xvg = XVG(array=data)
        xvg.write(filename)

        if plot:
            plt.plot(data[0,:] / 1000, data[1,:],  label=label, c='k')
            plt.ylabel('Deviations (nm)')
            plt.xlabel('Time (ns)')
            plt.legend()
            plt.show()

    if ndens: # Save the average histogram for each distance over time
        # bins = np.linspace(min_dev, max_dev, num=n_bins) # wider range to look at
        bins = np.linspace(-5, 5, num=n_bins)
        for frame in range(n_frames):
            histograms[frame,:], bin_edges[frame,:] = np.histogram(deviations[frame,:], bins=bins)

        data = np.array([np.mean(bin_edges[:,:-1], axis=0), np.mean(histograms, axis=0)])
        xvg = XVG(array=data)
        xvg.write(filename.split('.xvg')[0] + '_density.xvg')

        # # Some functionality to check overlapping distributions
        # h = np.zeros((n_frames, 10))
        # b = np.zeros((n_frames, 11))
        # for frame in range(n_frames):
            # h[frame,:], b[frame,:] = np.histogram(deviations[frame,:])
            # plt.hist(deviations[frame,:], color=c, alpha=0.1)

        # # Plotting the overlapping head group distributions
        # plt.plot(np.mean(b[:,:-1], axis=0), np.mean(h, axis=0), label=label, c=c)
        # plt.ylabel('Number density')
        # plt.xlabel('Deviations (nm)')
        # plt.legend()

        if plot:
            plt.plot(data[0,:], data[1,:],  label=label, c='k')
            plt.ylabel('Number density')
            plt.xlabel('Deviations (nm)')
            plt.legend()
            plt.show()

    return com, deviations, intersect_points


######################################################################################
################################# Generate data ######################################
######################################################################################

traj = md.load(xtc, top=gro)
traj_sliced = traj[start::step]

# Glycerol
selection = 'resname GLY'
filename = cwd + 'glycerol_time_series.xvg'
n_atoms_per_monomer = None
label = 'Glycerol'

print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
com_gly, deviations_gly, intersect_gly = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                                                             n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve, ts=ts, 
                                                             ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)


######################################################################################
############################### Testing on bilayer ###################################
######################################################################################

if test_bilayer:

    bilayer_traj = md.load('./test_bilayer/AA_bilayer.xtc', top='./test_bilayer/AA_bilayer.gro')
    traj_sliced = bilayer_traj[-1001::5]

    # Glycerol
    selection = 'resname GLY'
    filename = './test_bilayer/dev_gly.xvg'
    label = 'Glycerol'

    print('\nCalculating the bilayer deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_gly, deviations_gly, intersect_gly = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

    # Ions
    selection = 'resname BR'
    filename = './test_bilayer/dev_ion.xvg'
    label = 'Ions'

    print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_ion, deviations_ion, intersect_ion = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

    # Head group
    # selection = 'resname MOL and name C17 C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 C28 N3'
    selection = 'resname MOL and name C22 C23 C24 C25'
    filename = './test_bilayer/dev_head.xvg'
    label = 'Head groups'

    print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_head, deviations_head, intersect_head = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

    # Tail ends
    selection = 'resname MOL and name C C1 C44 C45'
    filename = './test_bilayer/dev_tail.xvg'
    label = 'Tail ends'

    print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_tail, deviations_tail, intersect_tail = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

    selection = 'resname MOL and name C C1'
    filename = './test_bilayer/dev_tail-left.xvg'
    label = 'Tail ends'

    print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_tail, deviations_tail, intersect_tail = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

    selection = 'resname MOL and name C44 C45'
    filename = './test_bilayer/dev_tail-right.xvg'
    label = 'Tail ends'

    print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
    com_tail, deviations_tail, intersect_tail = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=plane_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, plot=plot)

######################################################################################
########################## Plotting points and solutions #############################
######################################################################################

def Gyroid_plot(x,y,z, period=9.27): # <-- I am not sure how to change the period in plot_implicit, 
                                     #     but you can change it here
    N = 2*np.pi/period
    
    a = np.sin(N*x) * np.cos(N*y)
    b = np.sin(N*y) * np.cos(N*z)
    c = np.sin(N*z) * np.cos(N*x)
    
    return a + b + c

# Plotting the last frame
# r1 = 0
# r2 = int(470/2 - 1)
# r3 = -1

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# fig, ax = plot_implicit(Gyroid_plot, bbox=(0,9.27), a=0.5)
# ax.scatter3D(com_head[-1,r1:r2,0], com_head[-1,r1:r2,1], com_head[-1,r1:r2,2],
#              alpha=0.5,c='r')
# ax.scatter3D(com_gly[-1,r1:r2,0], com_gly[-1,r1:r2,1], com_gly[-1,r1:r2,2],
#              alpha=0.5,c='g')

# ax.scatter3D(intersect_head[-1,r1:r2,0], intersect_head[-1,r1:r2,1], intersect_head[-1,r1:r2,2],
#              alpha=0.5,c='r')

# for i,com in enumerate(com_head[-1,r1:r3,:]):
#     distance = np.vstack((com, intersect_head[-1,i,:]))
#     ax.plot3D(distance[:,0], distance[:,1], distance[:,2], c='red')

# for i,com in enumerate(com_ion[-1,r1:r3,:]):
#     distance = np.vstack((com, intersect_ion[-1,i,:]))
#     ax.plot3D(distance[:,0], distance[:,1], distance[:,2], c='blue')
    
# for i,com in enumerate(com_gly[-1,r1:r3,:]):
#     distance = np.vstack((com, intersect_gly[-1,i,:]))
#     ax.plot3D(distance[:,0], distance[:,1], distance[:,2], c='green')

# for i,com in enumerate(com_tail[-1,r1:r3,:]):
#     distance = np.vstack((com, intersect_tail[-1,i,:]))
#     ax.plot3D(distance[:,0], distance[:,1], distance[:,2], c='black')

# plt.show()

######################################################################################
############################### Testing the solver ###################################
######################################################################################
if test_solver:

    # Generate an artificial test point (this point is 'on' the surface with tolerance 10^-5)
    test_point_surf = np.array([8.73535354, 1.04444444, 3.32323232])
    print('Generated point on the surface: %s' %(test_point_surf))
    norm = Gyroid_grad(test_point_surf, 9.4) # project a normal from [8.73535354 1.04444444 3.32323232]
    fake_point = test_point_surf + 1.1 * norm # make up point 1.1 away from surface to test solver
    print('Distance away from surface: %s' %(1.1))

    params = {
        'period' : 9.4,
        'point'  : fake_point
    }

    guess_dist = 3
    guess_point = np.array([0, 7, 6])
    guess = (guess_dist, guess_point[0], guess_point[1], guess_point[2])

    sol = fsolve(eq_solve, guess, args=params)

    print('Not-so-good initial guess: %s distance from surface, %s point on surface' %(guess_dist, guess_point))
    print('Values of the equations at initial guess: %s %s %s %s' %(eq_solve(guess, params)[0], eq_solve(guess, params)[1], eq_solve(guess, params)[2], eq_solve(guess, params)[3]))
    print('Values of the equations at the solution: %s %s %s %s' %(eq_solve(sol, params)[0], eq_solve(sol, params)[1], eq_solve(sol, params)[2], eq_solve(sol, params)[3]))
    print('Solution found by fsolve: %s distance from surface, %s point on surface' %(sol[0], sol[1:]))