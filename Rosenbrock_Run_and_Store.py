#!/usr/bin/env python
"""
Sample code for sampling a Rosenbrock emcee. 
This test is for the lowest dimension
- Run
- Load
- Compare between the loaded and run factor
- Calculate 

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

import math 
import os
import pyfits
import os.path

import time

# take start time
start_time = datetime.now()

save_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
img_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'

test = False                 # Test = True or False
#test = True                   # Test = True or False
save_data = True              # save_data = False


def lnprob(x, ndim):    
    z = x[1:ndim:2]                                  # (2,4,6,8,10,......) X_2i
    y = x[0:ndim:2]                                  # (1,3,5,7,9,...)    X_2i-1      or y = z -1
    return -sum(100.0*(y**2-z)**2.0 + (y-1)**2.0)


def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data



M = 4   # amount of chains for the Gelman & Rubin analysis 

if test == False:
    # n=10, t=1000, 9 hours, 12 minutes , acccpetance = 0.169
    # n=50, t=1000, 4 hours, 41 minutes, accpetance = 0.157
    # n=100,t=1000, 2 hours 0 minutes, accpetance = 0.15
   ndims =     [ 100, 50, 10]
   thinnings =  [ int(100),int(50),int(10)]
   
   #thinnings =  [ int(1e3),int(5e3),int(1e4)]

   # steps = runlength 
   run_length  = int(1e5)
   half_length =   run_length
   total_length = 2*run_length
else:
    # n=10, t=1000, 1 minutes, acccpetance = 0.169
    # n=50, t=1000, 3 minutes, accpetance = 0.157
    # n=100, t=1000,8 minutes, accpetance = 0.15
   ndims =  [ 10,50,100]
   thinnings =  [ 10,10,10]

   # steps = runlength
   # n=10,t=200000,  is about  2 hours and  0 minutes,  rate
   # n=50,t=200000,  is about 4  hours and 41 minutes,   rate
   # n=100,t=200000, is about 9 hours and 13 minutes,            rate
   run_length  = int(1e3)
   half_length =   run_length
   total_length = 2*run_length

# dimension of data cube  (walkers  , length/thinning, ndim )   example:     #(250, 1000, 50)    = (#walkers, #steps, # parameters)
    #200 x 200 X 10  =  #walkers * #steps * #dim

# the initial distribution should be overly-dispersed


maximum_size = 536870912
print("\n ******************************************************************************")
print("This is the test run for the emcee using a Rosenbrock and write it to data files")


print("This program does not yet perform any data analysis")
# take start time
start_time = datetime.now()
for ndim_counter in range(0,len(ndims)):

    ndim = ndims[ndim_counter]

    print("\n The number of parameters: ",ndim)
    nwalkers = 10*ndim
    print(" The number of walkers: ",nwalkers)
    thinning = thinnings[ndim_counter]
    print(" The thinning is: ",thinning)
    print(" The runlength is: ",run_length," steps")
    print(" The chain-length is: ",total_length,"/",thinning,'=',total_length/thinning)

    # the size of the array for storage is  (#chains,#walkers, #run_length/thinning, #dimensions)

    array_size =  M*ndim*nwalkers*(total_length/thinning)

    print(" The number of chains", M, " and the array size is ", array_size)
    if array_size > maximum_size:
        print("This array is probably too large")
    data_set = np.ones([M,nwalkers,total_length/thinning, ndim])  # dimension  runs    * walkers*runlength/thinning *   #parameters
    print(" The shape of the data-cube is ", data_set.shape, "(#chains,#walkers, #run_length/thinning, #dimensions) \n")

    for m in range(0,M):
        print(" The run number is ", m);

	# Choose an initial set of positions for the walkers from different, overdispersed normal distributions

        if m == 0:
             sigma_values = 5
             p0 = sigma_values*np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim))

        if m == 1:
             sigma_values = 5
             p0 = sigma_values*np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim)) + np.ones(ndim * nwalkers).reshape((nwalkers, ndim))

        if m == 2:
             sigma_values = 5
             p0 = sigma_values*np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim)) - np.ones(ndim * nwalkers).reshape((nwalkers, ndim))

        if m == 3:
             sigma_values = 10
             p0 = sigma_values*np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim)) 


        # Initialize the sampler with the chosen specs.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4,args=[ndim])

	    # run there steps.
        sampler.run_mcmc(p0, total_length, thin=thinning)

        print(" There are ", len(p0) , "walkers to be intialized. These shape is: ",sampler.chain.shape )

     	# Print out the mean acceptance fraction. In general, acceptance_fraction has an entry for each walker so, in this case, it is a 250-dimensional vector.
        print(" Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

	   # Estimate the integrated autocorrelation time for the time series in each parameter.
       #print(" Autocorrelation time    :", sampler.get_autocorr_time())
       #print(" Max Autocorrelation time:", sampler.acor.max())
       # print(" Integrated Autocorrelation time:", emcee.autocorr.integrated_time(sampler.chain[0,:,:], axis=0) )
        print(sampler.chain.shape )    # gives (250, 1000, 50)  gives 250 vectors of 50 elements for a 'time' durations 1000 elements

    #    print("The shape of the chain obtain from sampler is: ", sampler.chain.shape)
        print(" Dimensions/parameters: ", sampler.chain.shape[2])
        print(" Walkers:               ", sampler.chain.shape[0])
        print(" Iterations:            ", sampler.chain.shape[1])
        print(" Shape sampler object: ", sampler.chain.shape)
    #    print("The shape of the data-set array is: ", sampler.chain.shape)
        print(" Shape data-cube is : ", data_set.shape, "(#chains,#walkers, #run_length/thinning, #dimensions)")
            # transfer the chain
            
        data_set[m,:,:,:] =  sampler.chain[:,:,:]
 
        print("------------------------------------------------------------------------------------------------ \n")

        
        
        # test correlation between different parameters by making an image
        fig = plt.figure(figsize=(8,6), dpi=300)
        plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.11)
        ax = fig.add_subplot(111)
        filename = "Rosenbrock_correlation_image__n="+ str(ndim)+"_j="+str(m) +"newest.png"
        completeName = os.path.join(img_path, filename)         
        reshaped = sampler.chain.reshape(total_length*nwalkers/thinning, ndim)
        plt.imshow(np.corrcoef(reshaped.T) )
        plt.savefig(filename)
        plt.close(fig)


        
         # store the data
    filename = "Rosenbrock_data_n="+ str(ndim)+"_j="+str(m) +"newest.txt"
    completeName = os.path.join(save_path, filename)
    write(data_set, completeName)

    #load data to control
    control_data = read(completeName)

    print(" Compare between the load and save data. First walker", np.mean(control_data[m,1,half_length/thinning:total_length/thinning,:]),"=", np.mean(sampler.chain[1,half_length/thinning:total_length/thinning,:]))
    print(" Compare between the load and save data. Last-step", np.mean(control_data[m,:,-1,:]),"=", np.mean(sampler.chain[:,-1,:]))
    print(" Compare between the load and save data. First parameter-step", np.mean(control_data[m,:,half_length/thinning:total_length/thinning,0]),"=", np.mean(sampler.chain[:,half_length/thinning:total_length/thinning,0]))

    if np.mean(control_data[m,1,half_length/thinning:total_length/thinning,:]) != np.mean(sampler.chain[1,half_length/thinning:total_length/thinning,:]):
                print("FAILED LOAD-STORE comparison 1")                   

    if np.mean(control_data[m,:,-1,:]) != np.mean(sampler.chain[:,-1,:]):
                print("FAILED LOAD-STORE comparison 2")                   

    if np.mean(control_data[m,:,half_length/thinning:total_length/thinning,0]) != np.mean(sampler.chain[:,half_length/thinning:total_length/thinning,0]):
                print("FAILED LOAD-STORE comparison 3 ")                   
    print(" Compare", np.mean(control_data[m,:,half_length/thinning:total_length/thinning,0]),"=", np.mean(sampler.chain[:,half_length/thinning:total_length/thinning,0]))
    print(" Compare", np.var(control_data[m,:,half_length/thinning:total_length/thinning,0]),"=", np.var(sampler.chain[:,half_length/thinning:total_length/thinning,0]))
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print("******************************************************************************************************* \n")

