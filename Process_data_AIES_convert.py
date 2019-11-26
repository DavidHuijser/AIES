#!/usr/bin/env python
"""
code for loading and plotting correlated multivariate Gaussian using emcee. 
To create graphs to be used in the paper.
Duration: 1:08:15.244036
Duration: 1:07:37.770178
"""

from __future__ import print_function
from __future__ import with_statement

import numpy as np
import matplotlib.pyplot as plt
import copy 
from datetime import datetime
import pylab

import matplotlib.patches as mpatches
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
plt.ion()

import pickle
import os.path

new_data = True

def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

# take start time
start_time = datetime.now()

M = 4   # amount of chains for the Gelman & Rubin analysis 

ndims =  [ 10,50, 100]
#thins =  [ 100,500,1000]
# the initial distribution should be overly-dispersed
load_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
save_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'


print("******************************************************************************")
print("This a program on convert the data to a format which is readable by R.")
print("Loading the data")
print("using NEW data")


for ndim_counter in range(0,len(ndims)):   
    # load data
    ndim = ndims[ndim_counter]
    #thinning = thins[ndim_counter]
    max_number_of_parameters_per_object  = ndims[ndim_counter]
    parameters_space_per_object  = ndims[ndim_counter]
    print("using Newest data")
    base_filename = 'Rosenbrock_data_n='
    filename = os.path.join(save_path, base_filename + str(ndim)+'_j=3newest.txt' )
    dat = read(filename)
    # get amount data point
    total_length = dat.shape[2]
    nwalkers = dat.shape[1]
    param = dat.shape[3]
    print(dat.shape)
    store_means = np.ones([M,int(0.5*total_length),param] )
    store_vars = np.ones([M,int(0.5*total_length),param] )
    
    for m in range(0,M):
       
           store_means[m,:,:]= np.mean(dat[m,:,int(0.5*total_length):total_length,:], axis=0)
           store_vars[m,:,:]= np.var(dat[m,:,int(0.5*total_length):total_length,:], axis=0)
          

    print("Length of chain:", total_length)
    print("Sample shape:", dat.shape)
    print("Number of walkers:", nwalkers)
    print("Parameters:", param)
    print(" Compare", np.mean(dat[M-1,:,int(0.5*total_length):total_length,0]))
    print(" Compare", np.var(dat[M-1,:,int(0.5*total_length):total_length,0]))




    filename = "Converted_Rosenbrock_data_means_n="+ str(ndim)+"_j="+str(3) +"NEWEST.txt"
    completeName = os.path.join(save_path, filename)   

    with file(completeName, 'w') as outfile:
        for slice_2d in store_means:
               np.savetxt(outfile, slice_2d)
    print("\n Print Mean X: ",np.mean(store_means[0:4,0,0]))
    print("\n Print Mean Y: ",np.mean(store_means[0,0:9,0]))
    print("\n Print Mean Z: ",np.mean(store_means[0,0,0:9]))


    filename = "Converted_Gaussian_data_var_n="+ str(ndim)+"_j="+str(3) +"NEWEST.txt"
    completeName = os.path.join(save_path, filename)   

    with file(completeName, 'w') as outfile:
        for slice_2d in store_vars:
               np.savetxt(outfile, slice_2d)
    print("\n Print Var X: ",np.mean(store_vars[0:4,0,0]))
    print("\n Print Var Y: ",np.mean(store_vars[0,0:9,0]))
    print("\n Print Var Z: ",np.mean(store_vars[0,0,0:9]))



end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("*******************************************************************************************************")
