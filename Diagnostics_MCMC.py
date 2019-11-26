#!/usr/bin/env python
"""
    Sample code for sampling a multivariate Gaussian using emcee.
    To create graphs to be used in the paper.
    """
from __future__ import print_function
import numpy as np
import emcee
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import pylab
import corner
import matplotlib.patches as mpatches
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
plt.ion()
import pickle
import pandas
import copy
import math
import os
import pyfits
import os.path
from MCMC_toolbox import gelman_rubin
from MCMC_toolbox import IAT


def read(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data

load_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
save_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'


# take start time
start_time = datetime.now()
# Info on the orginal run (universal for all runs and problems)
ndims =     [10, 50 , 100]
M = 4   # amount of chains for the Gelman & Rubin analysis
# steps = runlength

run_length  = int(1e5)
#half_length =   run_length
total_length = 2*run_length


# steps = runlength
# dimension of data cube  (walkers  , length/thinning, ndim )   example:     #(250, 1000, 50)    = (#walkers, #steps, # parameters)
#200 x 200 X 10  =  #walkers * #steps * #dim

# the initial distribution should be overly-dispersed
print("******************************************************************************")
print("This is the original analysis of the emcee using a correlated Gaussian ")
print("Gelman and Rubin Analysis for the AIES / EMCEE")
for ndim_counter in range(0,6):
    ndim = ndims[ndim_counter % 3]
    print("\n The number of parameters: ",ndim)
    if ndim_counter <3:
        filename = "Gaussian_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST.txt"
        print("GAUSSIAN")
    else:
        filename = "Rosenbrock_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST.txt"
        print("Rosenbrock")
    filename = os.path.join(save_path, filename)
    loaded_data_set = read(filename)
    #SHAPE:   (M , nwalkers, run_length/thinning, ndim)
    shape_data = np.shape(loaded_data_set)
    nwalkers =shape_data[1]
    thinning = total_length/shape_data[2]
    print(" The number of walkers ",nwalkers)
    print(" Thinning ",  thinning )
    print(" Parameters ", ndim)
    print(" Full length ",loaded_data_set.shape[2])
#   print(" Compare between the load and save data. First walker", np.mean(loaded_data_set[M-1,1,half_length/thinning:total_length/thinning,:]))
#   print(" Compare between the load and save data. Last-step", np.mean(loaded_data_set[M-1,:,-1,:]))
#   print(" Compare between the load and save data. First parameter-step", np.mean(loaded_data_set[M-1,:,half_length/thinning:total_length/thinning,0]))
    
#   print("Length of the entire run (including burn in)", loaded_data_set.shape[2])
    length = int(0.5*loaded_data_set.shape[2])

    # throw away burn-in
    chain = loaded_data_set[:,:,length:2*length,:]
    # take average over all walkers for GR
    y = np.mean(chain, axis=1)
    print("Mspfr of average over walkers")
    mspfr = gelman_rubin(y)

    means =  np.mean(loaded_data_set[3,:,length:2*length,0])
    vars =   np.var(loaded_data_set[3,:,length:2*length,0])
    print("Mspfr of variance over walkers")
    y = np.var(chain, axis=1)
    mspfr = gelman_rubin(y)

    print("\n IAT of average over walkers of the fourth chain")
    y = np.mean(chain, axis=1)
    IAT0 = IAT(y[3,:,0])
    print("IAT", IAT0)
    
    print("\n IAT of variance over walkers of the fourth chain")
    y = np.var(chain, axis=1)
    IAT1 = IAT(y[3,:,0])
    

    print("\n \n ")
    print("Calculate Mspfr over 4 walkers")
    # from walkers, with intial distributions: N(0,5),N(1,5),N(-1,5), N(0,10)
    test_set = np.array([chain[0,1,:,:], chain[1,0,:,:],chain[2,4,:,:],chain[3,8,:,:]])
#   print(np.shape(test_set))
    test = gelman_rubin(test_set)
    msprfs = np.zeros(4)
    for i in range(0,4):
       msprfs[i] = gelman_rubin(chain[i,0:4,:,:])
       
    print("Mean msprfs", np.mean(msprfs))
    print("Mean msprfs from 4th chain", msprfs[3])

    #SHAPE:   (M , nwalkers, run_length/thinning, ndim)
    print("Calculate IAT of individual walker (instead of average over walkers)")
    IATS = np.zeros(nwalkers)
    for i in range(0,nwalkers):
           IATS[i] = IAT(chain[3,i,:,0])
    print("Mean IAT", np.mean(IATS))
    print("==========================================================================")

    if ndim_counter <3:
        store_filename = "Gaussian_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST_Results.txt"
    else:
        store_filename = "Rosenbrock_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST_Results.txt"

    f = open(store_filename, "w")
    f.write( " Number of parameters is: "+ str(ndim) + "\n")      # str() converts to string
    f.write( " The thinning is: "+ str(thinning) + "\n")      # str() converts to string
    f.write( " The entire chainlength is: "+str(2*length) +  " steps \n \n" )      # str() converts to string
#    f.write( " The chain-length is: " + str(run_length)+"/"+ str(thinning) + '=' + str(run_length/thinning)+ "\n" )      # str() converts to string

    f.write( "Mean x1, second half of 4th chain over all walkers: " + str(means)+ " \n")
    f.write( "Variance x1, second half of 4th chain over all walkers: " + str(vars)+ "\n \n")

    f.write(" IAT of average over walkers of the 4th chain: "+str(IAT0) + "\n")
    f.write(" IAT of variance over walkers of the 4th chain: "+str(IAT1) + "\n")
    f.write(" average IAT of individual walker " + str(np.mean(IATS))  + "\n \n")

    f.write( "Mspfr of average over walkers using 4 chains: " + str (mspfr)+ "\n")
    f.write( "Mspfr of walkers per chain: \n")
    f.write( "Chain1:" + str(msprfs[0]) + "\n")
    f.write( "Chain2:" + str(msprfs[1]) + "\n")
    f.write( "Chain3:" + str(msprfs[2]) + "\n")
    f.write( "Chain4:" + str(msprfs[3]) + "\n \n")
    

    f.close()


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))






#   print(length)

#print("Test_data", np.shape(loaded_data_set))
#print("\n Print Mean X: ",loaded_data_set[0:4,0,0])
#print("\n Print Mean Y: ",loaded_data_set[0,0:9,0])
#print("\n Print Mean Z: ",loaded_data_set[0,0,0:10])


# chain = loaded_data_set[:,:,length:2*length,:]

#y = np.mean(chain, axis=1)

#print("\n Print first 4 chains: ",y[0:4,0,0])
#print("\n Print first 10 itterations: ",y[0,0:9,0])
#print("\n Print first 10 parameters: ",y[0,0,0:10])

#Y = y[0,:,0]
#IAT(Y)


stop

#print(np.shape(y))
#IAT(Y)

stop

#gelman_rubin(y)

#print(np.shape(y))
#Y = np.squeeze(y[0,:,1])
#print(np.shape(y))




