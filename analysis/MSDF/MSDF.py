#!/usr/bin/env python

######################################################################################
###################################### INPUTS ########################################
######################################################################################

# Choose whether to calculate MSDF on a glycerol-solvated membrane or a water-solvated membrane
glycerol_solv = False
water_solv = True
wat_pct = '21'

# Choose whether to calculate all heavy atom distances or COMs
heavy_atoms = True

# Set the current working directory and specify the trajectory and topology files
cwd = '/home/nate/Projects/backmapping/gyroid/backmapped-gyroid/AA-simulation/L9.3_nm/NPT_343K/xlink_wat_solv/wat_21/'        # current working directory where all files are located and should be saved
xtc = cwd + 'npt_long_2.xtc'                      # trajectory file
gro = cwd + 'solvated_water.gro'                      # topology file

# Select the frames over which to calculate the MSDF
start = 0                    # starting frame 
step = 2                         # frame step (will go from start to the final frame by this step)

# Set some parameters for output
ts = False                          # whether to calculate the distance from the minimal surface as a time series
ndens = False                        # whether to calculate the number density with respect to the minimal surface
n_bins = 200                        # number of bins for generating number density histograms
n_bootstraps = 200                  # number of bootstraps for the error on number density histograms
plot = False                        # whether to plot the resulting MSDF for each group
save_dist = True                    # whether to save the distances to a file

# Group specific parameters are in Gather Data section
#       --> save file name
#       --> selection to calculate MSDF for
#       --> number of atoms per monomer in selection
#       --> labels for plotting

######################################################################################
############################### Import Statements ####################################
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scipy.optimize import fsolve
from gromacs.formats import XVG
import scipy.stats as st
from MSDF_error import make_bootstraps, get_bootstrapped_histograms
import time 

######################################################################################
####################### Functions for performing analysis ############################
######################################################################################

# Timer class to store performance
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = time.perf_counter()
        self.saved_times = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and save the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        self.elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

    def save(self, name=None):
        """Save the elapsed_time to a dictionary for printing"""
        if self.elapsed_time is None:
            raise TimerError(f"Timer has no elapsed times. Use .stop() to calculate an elapsed time")
        
        if self.saved_times is None:
            self.saved_times = {}

        self.n_saved = len(self.saved_times)
        if name is None:
            name = 'ElapsedTime' + str(self.n_saved)

        self.saved_times[name] = self.elapsed_time

    def report(self, print_out=True, filename=None, title=None):
        """Print all saved times"""
        if self.saved_times is None:
            raise TimerError(f"Timer has no saved times. Use .save() to save an elapsed time")
            
        if title is None:
            title = 'Timer Report'

        lines = []
        lines.append('\n' + '-'*20 + ' ' + title + ' ' + '-'*20 + '\n\n')
        for t in self.saved_times:
            lines.append('{name:40}{elapsed_time:14.4f}\n'.format(name=t, elapsed_time=self.saved_times[t]))

        lines.append('\n' + '-'*(42+ len(title)) + '\n\n')

        if print_out:
            [print(line, end='') for line in lines]

        if filename is not None:
            f = open(filename, 'w')
            [f.write(line) for line in lines]


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


def calculate_deviation(traj, selection, label, n_atoms_per_monomer=None, c='black', eq_solve=Gyroid_solve, ts=True, ndens=True, filename='output.xvg', n_bins=50, n_bootstraps=1000, plot=False, verbose=False):

    # Initialize inner timer
    innerTimer = Timer()

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

    innerTimer.stop()
    innerTimer.save('initialization')
    innerTimer.start()

    loopTimer = Timer()

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


        loopTimer.stop()
        loopTimer.save('COM_calculation_' + str(frame))
        loopTimer.start()

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
        
        if frame % 20 == 0:
            print('Completed deviation calculations for frame %d out of %d!' %(frame, n_frames))

        loopTimer.stop()
        loopTimer.save('distance_calculation_' + str(frame))
        loopTimer.start()

    loopTimer.report(print_out=False, filename='loopTimer.txt', title='Loop Timing for ' + label)
    
    innerTimer.stop()
    innerTimer.save('distance_calculations')
    innerTimer.start()

    if ts: # Save the average deviation for each frame over time
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

    innerTimer.stop()
    innerTimer.save('saving_time_series')
    innerTimer.start()

    if ndens: # Save the average histogram for each distance over time
        bins = np.linspace(-5, 5, num=n_bins)

        # Calculate the mean bootstrapped histogram and standard deviation
        boot = make_bootstraps(deviations, n_bootstraps=n_bootstraps)
        histograms, bin_edges = get_bootstrapped_histograms(boot) # outputs the mean over the frames (shape is n_bootstraps by n_bins(-1))

        data = np.array([np.mean(bin_edges[:,:-1], axis=0), np.mean(histograms, axis=0), np.std(histograms, axis=0)]) # save the bootstrapped mean and bootstrapped stdev
        xvg = XVG(array=data, names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
        xvg.write(filename.split('.xvg')[0] + '_density.xvg')

        if plot:
            plt.plot(data[0,:], data[1,:],  label=label, c='k')
            plt.ylabel('Number density')
            plt.xlabel('Deviations (nm)')
            plt.legend()
            plt.show()

    innerTimer.stop()
    innerTimer.save('bootstrapping_saving_density')
    innerTimer.report(filename='innerTimer.txt', title='Inner Timer for ' + label)

    return com, deviations, intersect_points


def write_distances(distances, save_dist=True, filename=None):

    if save_dist:
        name = filename.split('.xvg')[0] + '.csv'
        print('Saving distances to {}'.format(name))
        np.savetxt(name, distances, delimiter=',')

    else: # Don't save distances
        print('Not saving distances...')


######################################################################################
################################# Generate data ######################################
######################################################################################

outerTimer = Timer()

traj = md.load(xtc, top=gro)
traj_sliced = traj[start::step]

outerTimer.stop()
outerTimer.save('read_trajectory')
outerTimer.start()

if glycerol_solv and water_solv:
    print('Cannot be both glycerol-solvated and water-solvated... Change line 8 or 9 and try again.')
    exit()

######################## GLYCEROL SOLVATED ########################

if glycerol_solv:

    n_atoms_per_monomer = None

    if not heavy_atoms: # normal COM group distances

        # Glycerol
        selection = 'resname GLY'
        filename = cwd + 'deviation_glycerol.xvg'
        label = 'Glycerol'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('glycerol_calculation')
        outerTimer.start()

        # Ions
        selection = 'resname BR'
        filename = cwd + 'deviation_ions.xvg'
        label = 'Ions'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('ions_calculation')
        outerTimer.start()

        # Head group
        filename = cwd + 'deviation_head.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47'
        label = 'Head groups'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                                                                        n_atoms_per_monomer=n_atoms_per_monomer, c='black', eq_solve=Gyroid_solve,
                                                                        ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_groups_calculation')
        outerTimer.start()

        # Tail ends
        filename = cwd + 'deviation_tail-ends.xvg'
        selection = 'resname MOL and name C C1 C44 C45'
        label = 'Tail ends'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('tail_ends_calculation')
        outerTimer.start()

        # Head groups - inside
        filename = cwd + 'deviation_head-in.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47 and index 0 to 33048'
        label = 'Head groups on one side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                            n_atoms_per_monomer=n_atoms_per_monomer, c='blue',
                            eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_group_in_calculation')
        outerTimer.start()

        # Head groups - outside
        filename = cwd + 'deviation_head-out.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47 and index 33049 to 66096'
        label = 'Head groups on other side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                            n_atoms_per_monomer=n_atoms_per_monomer, c='red',
                            eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, 
                            n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_group_out_calculation')
        outerTimer.start()

        # Full tails
        filename = cwd + 'deviation_tails.xvg'
        selection = 'resname MOL and name C C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38 C39 C40 C41 C42 C43 C44 C45'
        label = 'Tails'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('full_tails_calculation')
        outerTimer.report(filename='gly_solv_COM_timing.txt', title='Timing for glycerol-solvated, COM groups')

    else: # calculate distances for all heavy atoms

        # Glycerol heavy atoms
        heavy = []
        for res in traj_sliced.top.residues:
            if res.name == 'GLY':
                i = 0
                for atom in res.atoms:
                    if not atom.element.name == 'hydrogen':

                        if len(heavy) < 6: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                            heavy.append([atom.index])  
                        else: # append to appropriate list
                            heavy[i].append(atom.index) 
                            i += 1                     

                             
        for a, atom in enumerate(heavy):
            filename = cwd + 'GLY_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'GLY_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('glycerol' + str(a) + '_calculation')
            outerTimer.start()

        # Monomer heavy atoms (all)
        heavy = []
        for res in traj_sliced.top.residues:
            if res.name == 'MOL':
                i = 0
                for atom in res.atoms:
                    if not atom.element.name == 'hydrogen':

                        if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                            heavy.append([atom.index])  
                        else: # append to appropriate list
                            heavy[i].append(atom.index) 
                            i += 1        
                             
        for a, atom in enumerate(heavy):
            filename = cwd + 'MOL_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'MOL_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('monomers_full' + str(a) + '_calculation')
            outerTimer.start()

        # Monomer heavy atoms (inside)
        heavy = []
        for res in traj_sliced.top.residues:
            if res.name == 'MOL':
                i = 0
                for atom in res.atoms:
                    if not atom.element.name == 'hydrogen' and atom.index < 33048:

                        if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                            heavy.append([atom.index])  
                        else: # append to appropriate list
                            heavy[i].append(atom.index) 
                            i += 1        
                             
        for a, atom in enumerate(heavy):
            filename = cwd + 'MOL_in_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'MOL_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('monomers_inside' + str(a) + '_calculation')
            outerTimer.start()

        # Monomer heavy atoms (outside)
        heavy = []
        for res in traj_sliced.top.residues:
            if res.name == 'MOL':
                i = 0
                for atom in res.atoms:
                    if not atom.element.name == 'hydrogen' and atom.index >= 33048:

                        if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                            heavy.append([atom.index])  
                        else: # append to appropriate list
                            heavy[i].append(atom.index) 
                            i += 1        
                             
        for a, atom in enumerate(heavy):
            filename = cwd + 'MOL_out_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'MOL_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('monomers_outside' + str(a) + '_calculation')
            outerTimer.start()

        # Ion heavy atoms
        heavy = []
        for res in traj_sliced.top.residues:
            if res.name == 'BR':
                i = 0
                for atom in res.atoms:
                    if not atom.element.name == 'hydrogen':

                        if len(heavy) < 1: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                            heavy.append([atom.index])  
                        else: # append to appropriate list
                            heavy[i].append(atom.index) 
                            i += 1                     

                             
        for a, atom in enumerate(heavy):
            filename = cwd + 'ION_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'ION_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('ions' + str(a) + '_calculation')
            outerTimer.start()
        
        outerTimer.report(filename='gly_solv_heavy_timing.txt', title='Timing for glycerol-solvated, heavy atoms')



######################## WATER SOLVATED ########################

if water_solv:

    if not heavy_atoms: # normal COM group distances

        # Water
        selection = 'resname HOH and name O'
        filename = cwd + 'deviation_water.xvg'
        label = 'Water'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('water_calculation')
        outerTimer.start()

        # Ions
        selection = 'resname BR'
        filename = cwd + 'deviation_ions.xvg'
        label = 'Ions'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('ions_calculation')
        outerTimer.start()

        # Head group
        filename = cwd + 'deviation_head.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47'
        n_atoms_per_monomer = 16
        label = 'Head groups'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                                                                        n_atoms_per_monomer=n_atoms_per_monomer, c='black', eq_solve=Gyroid_solve,
                                                                        ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_groups_calculation')
        outerTimer.start()

        # Tail ends
        filename = cwd + 'deviation_tail-ends.xvg'
        selection = 'resname MOL and name C C1 C44 C45'
        n_atoms_per_monomer = 4
        label = 'Tail ends'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('tail_ends_calculation')
        outerTimer.start()

        # Head groups - inside
        filename = cwd + 'deviation_head-in.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47 and index 0 to 33048'
        n_atoms_per_monomer = 16
        label = 'Head groups on one side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                            n_atoms_per_monomer=n_atoms_per_monomer, c='blue',
                            eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_groups_in_calculation')
        outerTimer.start()

        # Head groups - outside
        filename = cwd + 'deviation_head-out.xvg'
        selection = 'resname MOL and name C18 N C19 C20 N1 C21 C22 C23 C24 C25 C26 N2 C27 N3 C46 C47 and index 33049 to 66096'
        n_atoms_per_monomer = 16
        label = 'Head groups on other side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label, 
                            n_atoms_per_monomer=n_atoms_per_monomer, c='red',
                            eq_solve=Gyroid_solve, ts=ts, ndens=ndens, filename=filename, 
                            n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('head_groups_out_calculation')
        outerTimer.start()

        # Full tails - inside
        filename = cwd + 'deviation_tails-in.xvg'
        selection = 'resname MOL and name C C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38 C39 C40 C41 C42 C43 C44 C45 and index 0 to 33048'
        n_atoms_per_monomer = 36
        label = 'Tails on one side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('full_tails_in_calculation')
        outerTimer.start()

        # Full tails - outside
        filename = cwd + 'deviation_tails-out.xvg'
        selection = 'resname MOL and name C C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38 C39 C40 C41 C42 C43 C44 C45 and index 33049 to 66096'
        n_atoms_per_monomer = 36
        label = 'Tails on other side'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('full_tails_out_calculation')
        outerTimer.start()

        # Full tails
        filename = cwd + 'deviation_tails.xvg'
        selection = 'resname MOL and name C C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38 C39 C40 C41 C42 C43 C44 C45'
        n_atoms_per_monomer = 36
        label = 'Tails'

        print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), selection))
        com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                            n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                            ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        write_distances(deviations, save_dist=save_dist, filename=filename)

        outerTimer.stop()
        outerTimer.save('full_tails_calculation')
        outerTimer.start()

    else: # calculate distances for all heavy atoms

        n_atoms_per_monomer = 1

        # # Water heavy atoms
        # heavy = []
        # for res in traj_sliced.top.residues:
        #     if res.name == 'HOH':
        #         i = 0
        #         for atom in res.atoms:
        #             if not atom.element.name == 'hydrogen':

        #                 if len(heavy) < 1: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
        #                     heavy.append([atom.index])  
        #                 else: # append to appropriate list
        #                     heavy[i].append(atom.index) 
        #                     i += 1                     

                             
        # for a, atom in enumerate(heavy):
        #     filename = cwd + 'SOL_heavy' + str(a) + '.xvg'
        #     selection = 'index'
        #     for a1 in atom:
        #         selection += ' ' + str(a1)
        #     label = 'SOL_heavy' + str(a)

        #     print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
        #     com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
        #                         n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
        #                         ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        #     write_distances(deviations, save_dist=save_dist, filename=filename)

        #     outerTimer.stop()
        #     outerTimer.save('water' + str(a) + '_calculations')
        #     outerTimer.start()             
                             
        # # Monomer heavy atoms in crosslinked membrane (full)
        # heavy = []
        # i = 0
        # for atom in traj_sliced.top.residue(0).atoms:
        #     if not atom.element.name == 'hydrogen':

        #         if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
        #             heavy.append([atom.index])  
        #         elif i < 52: # append to appropriate list for each monomer
        #             heavy[i].append(atom.index) 
        #             i += 1
        #         else: # when we loop to next monomer, restart indexing
        #             i = 0
        #             heavy[i].append(atom.index)
        #             i += 1

        # for a, atom in enumerate(heavy):
        #     filename = cwd + 'MOL_heavy' + str(a) + '.xvg'
        #     selection = 'index'
        #     for a1 in atom:
        #         selection += ' ' + str(a1)
        #     label = 'MOL_heavy' + str(a)

        #     print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
        #     com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
        #                         n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
        #                         ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        #     write_distances(deviations, save_dist=save_dist, filename=filename)

        #     outerTimer.stop()
        #     outerTimer.save('monomers' + str(a) + '_calculation')
        #     outerTimer.start()

        # Monomer heavy atoms in crosslinked membrane (inside)
        heavy = []
        i = 0
        for atom in traj_sliced.top.residue(0).atoms:
            if not atom.element.name == 'hydrogen' and atom.index < 33048:

                if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                    heavy.append([atom.index])  
                elif i < 52: # append to appropriate list for each monomer
                    heavy[i].append(atom.index) 
                    i += 1
                else: # when we loop to next monomer, restart indexing
                    i = 0
                    heavy[i].append(atom.index)
                    i += 1

        for a, atom in enumerate(heavy):
            filename = cwd + 'MOL_in_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'MOL_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('monomers_in' + str(a) + '_calculation')
            outerTimer.start()

        # Monomer heavy atoms in crosslinked membrane (outside)
        heavy = []
        i = 0
        for atom in traj_sliced.top.residue(0).atoms:
            if not atom.element.name == 'hydrogen' and atom.index >= 33048:

                if len(heavy) < 52: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
                    heavy.append([atom.index])  
                elif i < 52: # append to appropriate list for each monomer
                    heavy[i].append(atom.index) 
                    i += 1
                else: # when we loop to next monomer, restart indexing
                    i = 0
                    heavy[i].append(atom.index)
                    i += 1

        for a, atom in enumerate(heavy):
            filename = cwd + 'MOL_out_heavy' + str(a) + '.xvg'
            selection = 'index'
            for a1 in atom:
                selection += ' ' + str(a1)
            label = 'MOL_heavy' + str(a)

            print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
            com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
                                n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
                                ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
            write_distances(deviations, save_dist=save_dist, filename=filename)

            outerTimer.stop()
            outerTimer.save('monomers_out' + str(a) + '_calculation')
            outerTimer.start()

        # # Ion heavy atoms
        # heavy = []
        # for res in traj_sliced.top.residues:
        #     if res.name == 'BR':
        #         i = 0
        #         for atom in res.atoms:
        #             if not atom.element.name == 'hydrogen':

        #                 if len(heavy) < 1: # create lists for each heavy atom within heavy (requires input of number of heavy atoms)
        #                     heavy.append([atom.index])  
        #                 else: # append to appropriate list
        #                     heavy[i].append(atom.index) 
        #                     i += 1                     

                             
        # for a, atom in enumerate(heavy):
        #     filename = cwd + 'ION_heavy' + str(a) + '.xvg'
        #     selection = 'index'
        #     for a1 in atom:
        #         selection += ' ' + str(a1)
        #     label = 'ION_heavy' + str(a)

        #     print('\nCalculating the deviation from %.2f ps to %.2f ps over %d frames for \'%s\'...\n' %(traj_sliced.time[0], traj_sliced.time[-1], len(traj_sliced), label))
        #     com, deviations, intersect = calculate_deviation(traj=traj_sliced, selection=selection, label=label,
        #                         n_atoms_per_monomer=n_atoms_per_monomer, eq_solve=Gyroid_solve,
        #                         ts=ts, ndens=ndens, filename=filename, n_bins=n_bins, n_bootstraps=n_bootstraps, plot=plot)
        #     write_distances(deviations, save_dist=save_dist, filename=filename)

        #     outerTimer.stop()
        #     outerTimer.save('ions' + str(a) + '_calculation')
        #     outerTimer.start()
        
        outerTimer.report(filename='wat_solv_' + wat_pct + '_heavy_timing.txt', title='Timing for water-solvated ' + wat_pct + ', heavy atoms')
