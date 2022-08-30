#!/usr/bin/env python

# Import packages
import numpy as np
import pandas as pd
import random
from scipy.optimize import fsolve
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import argparse
from sklearn.neighbors import KernelDensity
# from KDEpy import FFTKDE
from sklearn.model_selection import GridSearchCV, KFold

##################################################################################################################
####################################### FUNCTIONS FOR CALCULATIONS ###############################################
##################################################################################################################

# Define the triply periodic minimal surface functions and their gradients
def SchwarzD(X, period):

    N = 2*np.pi/period
    
    a = np.sin(N*X[0]) * np.sin(N*X[1]) * np.sin(N*X[2])
    b = np.sin(N*X[0]) * np.cos(N*X[1]) * np.cos(N*X[2])
    c = np.cos(N*X[0]) * np.sin(N*X[1]) * np.cos(N*X[2])
    d = np.cos(N*X[0]) * np.cos(N*X[1]) * np.sin(N*X[2])
    
    return a + b + c + d

def Gyroid(X,period):
    
    N = 2*np.pi/period
    
    a = np.sin(N*X[0]) * np.cos(N*X[1])
    b = np.sin(N*X[1]) * np.cos(N*X[2])
    c = np.sin(N*X[2]) * np.cos(N*X[0])
    
    return a + b + c

def SchwarzD_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a = N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z)
    b = N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z) - N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z) - N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z)
    c = N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z) - N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Gyroid_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a =  N*np.cos(N*x)*np.cos(N*y) - N*np.sin(N*x)*np.sin(N*z)
    b = -N*np.sin(N*y)*np.sin(N*x) + N*np.cos(N*y)*np.cos(N*z)
    c = -N*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*z)*np.cos(N*x)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Primitive(X, period):
    
    N = 2*np.pi/period
    
    a = np.cos(N*X[0]) + np.cos(N*X[1]) + np.cos(N*X[2])
    
    return a

def Primitive_grad(v, period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a = -N*np.sin(N*x) 
    b = -N*np.sin(N*y) 
    c = -N*np.sin(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))


# Functions for solving the equations

def P_gyroid(t,X,period=9.4): # return point extended along the normal from gyroid surface
    n = Gyroid_grad(X,period)
    
    return X + t*n

def P_schwarz(t,X,period=9.4): # return point extended along the normal from schwarz surface
    n = SchwarzD_grad(X,period)
    
    return X + t*n

def P_primitive(t,X,period=9.4): # return point extended along the normal from primitive surface
    n = Primitive_grad(X,period)
    
    return X + t*n

# functions for numerical solver that outputs a function of the distance projected along normal
def Gyroid_eq(X,period=9.4): 

    n = Gyroid_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.sin(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) + np.sin(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.sin(N * (X[2] + t*n[2])) * np.cos(N * (X[0] + t*n[0]))

def SchwarzD_eq(X,period=9.4):
    
    n = SchwarzD_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.sin(N * (X[0] + t*n[0])) * np.sin(N * (X[1] + t*n[1])) * np.sin(N * (X[2] + t*n[2])) + np.sin(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.cos(N * (X[0] + t*n[0])) * np.sin(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.cos(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) * np.sin(N * (X[2] + t*n[2]))

def Primitive_eq(X,period=9.4):

    n = Primitive_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.cos(N * (X[0] + t*n[0])) + np.cos(N * (X[1] + t*n[1])) + np.cos(N * (X[2] + t*n[2]))


# Main function for generating the distance distributions
def surface2surface(structure,struct='gyroid',guess=4.5,box=9.4,period=9.4,sample=10,restrictions=True):
    
    # some hard coded constants
    short_dist_tol = 0.01
    small_tol = 0.01

    # intialize all variables
    distribution = []
    neg = []
    small = []
    big = []
    good = []

    # define necessary functions for the chosen structure
    if struct == 'gyroid':
        P = P_gyroid
        solve_eq = Gyroid_eq
    elif struct == 'schwarzD':
        P = P_schwarz
        solve_eq = SchwarzD_eq
    elif struct == 'primitive':
        P = P_primitive
        solve_eq = Primitive_eq

    # generate distribution of surface to surface distances
    while len(distribution) < sample:

        p = random.randint(0, structure.shape[0] - 1) # choose a random point on the discretized surface
        point = structure[p,:]
            
        # find the point actually on the surface
        short_dist = fsolve(solve_eq(point,period), short_dist_tol)
        new_point = P(short_dist, point, period=period)

        # solve for the distance from the new point to another point on the minimal surface
        sol = fsolve(solve_eq(new_point,period), guess)
        
        if restrictions:
            if abs(sol) < box and abs(sol) > small_tol: # only save reasonable values
                distribution.append(abs(sol))
        else:
            distribution.append(sol) # save every value

        # keep track of the abnormalities
        if abs(sol) < box and abs(sol) > small_tol:
            good.append(new_point)
        if sol < 0 and abs(sol) > small_tol:
            neg.append(new_point)
            point2 = P(0.001,new_point,period=period)
            sol2 = fsolve(solve_eq(point2,period), guess)
            print('Point %s' %(new_point))
            print('\tOriginal: %.4f' %(sol) )
            print('\tResolved: %.4f' %(sol2) )
        if abs(sol) < small_tol:
            small.append(new_point)
        if abs(sol) > box:
            big.append(new_point)
                
    print('\nReasonable solutions for %s: %d' %(struct, len(good)))
    print('Negative solutions for %s: %d' %(struct, len(neg)))
    print('Solutions < %s nm for %s: %d' %(small_tol, struct, len(small)))
    print('Solutions outside the box for %s: %d' %(struct, len(big)))
    
    return distribution

def brentq_solver(structure,struct='gyroid',box=9.4,period=9.4,sample=10):
    
    # some hard coded constants
    short_dist_tol = 0.01
    small_tol = 0.01
    upper = box
    lower = small_tol
    checked = False

    # intialize all variables
    distribution = []
    neg = []
    small = []
    big = []
    good = []

    # define necessary functions for the chosen structure
    if struct == 'gyroid':
        P = P_gyroid
        solve_eq = Gyroid_eq
    elif struct == 'schwarzD':
        P = P_schwarz
        solve_eq = SchwarzD_eq
    elif struct == 'primitive':
        P = P_primitive
        solve_eq = Primitive_eq

    # generate distribution of surface to surface distances
    while len(distribution) < sample:

        if not checked:
            p = random.randint(0, structure.shape[0] - 1) # choose a random point on the discretized surface
            point = structure[p,:]
            
            # find the point actually on the surface
            short_dist = fsolve(solve_eq(point,period), short_dist_tol)
            new_point = P(short_dist, point, period=period)

        # solve for the distance from the new point to another point on the minimal surface
        dt = 0.01
        a = None; b = None
        n_sol = False; p_sol = False
        for t in np.arange(lower,upper,dt):
            if t < -2*small_tol or t > 2*small_tol:
                f = solve_eq(new_point, period=period)(t)
                if f < 0:
                    a = t
                    n_sol = True
                if f > 0:
                    b = t
                    p_sol = True
                if n_sol and p_sol:
                   break

        if not n_sol or not p_sol:
            if not checked:
                # print('Could not find a bracket for the solution... Changing the search range.')
                lower = -box
                upper = -small_tol
                checked = True
            else:
                print('Could not find a bracket for the solution... Exiting.')
                exit()
        elif checked:
            lower = small_tol
            upper = box
            checked = False
            
        if not checked:
            sol = brentq(solve_eq(new_point,period), a=a, b=b)

            distribution.append(abs(sol)) # save every value

            # keep track of the abnormalities
            if abs(sol) < box and abs(sol) > small_tol:
                good.append(new_point)
            if sol < 0 and abs(sol) > small_tol:
                neg.append(new_point)
            if abs(sol) < small_tol:
                small.append(new_point)
            if abs(sol) > box:
                big.append(new_point)
                
    print('\nReasonable solutions for %s: %d' %(struct, len(good)))
    print('Negative solutions for %s: %d' %(struct, len(neg)))
    print('Solutions < %s nm for %s: %d' %(small_tol, struct, len(small)))
    print('Solutions outside the box for %s: %d' %(struct, len(big)))
    
    return distribution



##################################################################################################################
############################################ INPUT PARAMETERS ####################################################
##################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-s','--struct',nargs='+',
                    help='structure to generate distribution for')
parser.add_argument('-n','--sample',type=int,
                    help='number of samples in each distribution')
parser.add_argument('-r','--restrict',action='store_true',
                    help='only save reasonable values, default = False')
parser.add_argument('-o','--output',default='output.png',
                    help='name of the output figure file')
args = parser.parse_args()

# Parse arguments for which distributions to generate
sample = args.sample
restrictions = args.restrict
if 'gyroid' in args.struct:
    gyroid = True
else:
    gyroid = False

if 'schwarz' in args.struct:
    schwarz = True
else:
    schwarz = False

if 'primitive' in args.struct:
    primitive = True
else:
    primitive = False


# Some hard coded parameters to use in all distributions
n = 100                 # grid size for discretized structure
box = 9.4               # box size in nm
period = box            # period of the minimal surface in nm
struct_tol = 0.01       # tolerance for locating discretized points on the surface
guess = 4.6             # initial guess for the numerical solvers --> corresponds to the bilayer distance + expected pore size from MD and experiment
# random.seed(123)      # uncomment if you want to set a random seed for reproducibility


##################################################################################################################
######################################### GENERATE DISTRIBUTIONS #################################################
##################################################################################################################

if gyroid:

    # Generate the structure
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    C = Gyroid(X, period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -struct_tol < C[i,j,k] < struct_tol:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]

    # Generate the distribution
    # dist_gyroid = surface2surface(structure,struct='gyroid',guess=guess,box=box,period=period,sample=sample,restrictions=restrictions)
    dist_gyroid = brentq_solver(structure,struct='gyroid',box=box,period=period,sample=sample)
    hist_gyroid = pd.DataFrame(dist_gyroid,columns=['Gyroid'])
    np.savetxt(args.output + '_gyroid_raw.txt', dist_gyroid, header='Gyroid pore-to-pore distances (nm)')

if schwarz:

    # Generate the structure
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    C = SchwarzD(X,period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -struct_tol < C[i,j,k] < struct_tol:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]

    # Generate the distribution
    #dist_schwarzD = surface2surface(structure,struct='schwarzD',guess=guess,box=box,period=period,sample=sample,restrictions=restrictions)
    dist_schwarzD = brentq_solver(structure,struct='schwarzD',box=box,period=period,sample=sample)
    hist_schwarzD = pd.DataFrame(dist_schwarzD,columns=['SchwarzD'])
    np.savetxt(args.output + '_schwarz_raw.txt', dist_schwarzD, header='SchwarzD pore-to-pore distances (nm)')



if primitive:

    # Generate the structure
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    C = Primitive(X, period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -struct_tol < C[i,j,k] < struct_tol:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]

    # Generate the distribution
    # dist_primitive = surface2surface(structure,struct='primitive',guess=guess,box=box,period=period,sample=sample,restrictions=restrictions)
    dist_primitive = brentq_solver(structure,struct='primitive',box=box,period=period,sample=sample)
    hist_primitive = pd.DataFrame(dist_primitive,columns=['Primitive'])
    np.savetxt(args.output + '_primitive_raw.txt', dist_primitive, header='Primitive pore-to-pore distances (nm)')


##################################################################################################################
######################################### PLOT DISTRIBUTIONS #####################################################
##################################################################################################################

# Plot histograms with matplotlib
print()
fig, ax = plt.subplots(1,1, figsize=(10,8))
if gyroid:

# No kernel density
    # bins = np.linspace(0,10,50)
    # ax.hist(hist_gyroid['Gyroid'], bins=bins, alpha=0.5, label='Gyroid', density=True)

# Manual bandwidth choice with sklearn
    # X_plot = np.linspace(0,10,1000)
    # reshaped = np.array(hist_gyroid['Gyroid']).reshape(-1,1)
    # kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(reshaped)
    # log_dens = kde.score_samples(X_plot.reshape(-1,1))
    # ax.plot(X_plot, np.exp(log_dens), label='Gyroid')

# K-fold cross validation for bandwidth choice with sklearn
    X_plot = np.linspace(0,10,1000)
    reshaped = np.array(hist_gyroid['Gyroid']).reshape(-1,1)

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(n_splits=10))
    grid.fit(reshaped)
    bw = grid.best_params_['bandwidth']
    print('Best bandwidth for gyroid:', bw)

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(reshaped)
    log_dens = kde.score_samples(X_plot.reshape(-1,1))
    ax.plot(X_plot, np.exp(log_dens), label='Gyroid')

    # Save data
    header = 'Geometric sampling of pore-to-pore distances for the Gyroid phase'
    np.savetxt(args.output + '_gyroid.txt', np.array([X_plot, np.exp(log_dens)]), header=header)

# Improved Sheather Jones bandwidth choice with KDEpy
    # reshaped = np.array(hist_gyroid['Gyroid']).reshape(-1,1)
    # x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(reshaped).evaluate()
    # ax.plot(x, y, label='Gyroid')

if schwarz:

# No kernel density
#     bins = np.linspace(0,10,50)
#     ax.hist(hist_schwarzD['SchwarzD'], bins=bins,
#             alpha=0.5, label='Schwarz Diamond')

# Manual bandwidth choice with sklearn
    # X_plot = np.linspace(0,10,1000)
    # reshaped = np.array(hist_schwarzD['SchwarzD']).reshape(-1,1)
    # kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(reshaped)
    # log_dens = kde.score_samples(X_plot.reshape(-1,1))
    # ax.plot(X_plot, np.exp(log_dens), label='Schwarz Diamond')

# K-fold cross validation for bandwidth choice with sklearn
    X_plot = np.linspace(0,10,1000)
    reshaped = np.array(hist_schwarzD['SchwarzD']).reshape(-1,1)

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(n_splits=10))
    grid.fit(reshaped)
    bw = grid.best_params_['bandwidth']
    print('Best bandwidth for schwarzD:', bw)

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(reshaped)
    log_dens = kde.score_samples(X_plot.reshape(-1,1))
    ax.plot(X_plot, np.exp(log_dens), label='SchwarzD')

    # Save data
    header = 'Geometric sampling of pore-to-pore distances for the Schwarz Diamond phase'
    np.savetxt(args.output + '_schwarz.txt', np.array([X_plot, np.exp(log_dens)]), header=header)

# Improved Sheather Jones bandwidth choice with KDEpy
    # reshaped = np.array(hist_schwarzD['SchwarzD']).reshape(-1,1)
    # x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(reshaped).evaluate()
    # ax.plot(x, y, label='SchwarzD')

if primitive:

# No kernel density
#     bins = np.linspace(0,10,50)
#     ax.hist(hist_primitive['Primitive'], bins=bins,
#        alpha=0.5, label='Primitive')

# Manual bandwidth choice with sklearn
    # X_plot = np.linspace(0,10,1000)
    # reshaped = np.array(hist_primitive['Primitive']).reshape(-1,1)
    # kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(reshaped)
    # log_dens = kde.score_samples(X_plot.reshape(-1,1))
    # ax.plot(X_plot, np.exp(log_dens), label='Primitive')

# K-fold cross validation for bandwidth choice with sklearn
    X_plot = np.linspace(0,10,1000)
    reshaped = np.array(hist_primitive['Primitive']).reshape(-1,1)

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(n_splits=10))
    grid.fit(reshaped)
    bw = grid.best_params_['bandwidth']
    print('Best bandwidth for primitive:', bw)

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(reshaped)
    log_dens = kde.score_samples(X_plot.reshape(-1,1))
    ax.plot(X_plot, np.exp(log_dens), label='Primitive')

    # Save data
    header = 'Geometric sampling of pore-to-pore distances for the Primitive phase'
    np.savetxt(args.output + '_primitive.txt', np.array([X_plot, np.exp(log_dens)]), header=header)

# Improved Sheather Jones bandwidth choice with KDEpy
    # reshaped = np.array(hist_primitive['Primitive']).reshape(-1,1)
    # x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(reshaped).evaluate()
    # ax.plot(x, y, label='Primitive')


# Add the bilayer thickness + pore size line
ax.axvline(4.6, color='black', linestyle='dashed', label='Expected pore-to-pore distance')
ax.axvspan(4.6 - 0.2, 4.6 + 0.2, color='gray', alpha=0.5)

# Some formatting
ax.set_xlim(0,10)
ax.set_xticks(np.arange(0,11,1))
# ax.set_ylim(0,0.5)
ax.set_xlabel('distance (nm)',fontsize='large')
ax.set_ylabel('probability density',fontsize='large')
ax.legend(fontsize='x-large',loc=1)
title = 'Theoretical pore center to pore center distances for BCC structures'
# ax.set_title(title)
plt.show()
# fig.savefig(args.output + '.png')

