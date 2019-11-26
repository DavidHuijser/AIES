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
import math

import math 
import os
import os.path



#save_data = True
#save_data = False

save_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
img_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'

def lnprob(x, alpha):
   # previously made a mistake:     beta =  1/(1.0 - alpha**2)**0.5
   # previously made a mistake:     ln_prob =   ln_prob - sum(0.5*diff**2/beta**2)
   beta =  (1.0 - alpha**2)**0.5
   ln_prob =   -0.5*x[0]**2
   diff = x[1:-1] - alpha*x[0:-2]
   ln_prob =   ln_prob - sum(0.5*diff**2)/(beta*beta) - 0.5*((x[len(x)-1] - alpha*x[len(x)-2])**2)/(beta*beta)
   return ln_prob

def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

M = 4  # amount of chains for the Gelman & Rubin analysis

# n=10,t=1000,  is about 1 minutes
# n=50,t=1000,  is about 10 minutes
# n=100,t=1000, is about 20 minutes

# n=10,t=200000,  is about 1 hours and 45 minutes, 0.41 rate
# n=50,t=200000,  is about 2 hours and 30 minutes, 0.18 rate
# n=100,t=200000, is about 2 hours and 55 minutes, 0.13 rate


ndims =     [ 10,50, 100]
#thinnings =  [ 10,50,100]
thinnings =  [ int(10),int(50),int(100)]


#ndims =     [ 100,50, 10]
#thinnings =  [ 10,50,100]
#thinnings =  [ int(100),int(50),int(10)]

alpha =  0.9


# steps = runlength 
run_length  = 1e5
half_length =   int(run_length)
total_length = int(2*run_length)

# dimension of data cube  (walkers  , length/thinning, ndim )   example:     #(250, 1000, 50)    = (#walkers, #steps, # parameters)
                        #200 x 200 X 10  =  #walkers * #steps * #dim           

print("\n ******************************************************************************")
print("This is the original analysis of the emcee using a correlated Gaussian and write it to data files")
print("This program does to perform any data analysis")
for ndim_counter in range(0,len(ndims)):
    ndim = ndims[ndim_counter]
    print("\n The number of parameters: ",ndim)
    nwalkers = 2*ndim
    print(" The number of walkers: ",nwalkers)
    thinning = thinnings[ndim_counter]
    print(" The thinning is: ",thinning)
    print(" The runlength is: ",run_length," steps")
    print(" The chain-length is: ",run_length,"/",thinning,'=',run_length/thinning)

    # the size of the array for storage is  run_length*2*ndinm*ndim/thinning
    array_size = M*nwalkers*ndim*2*total_length/thinning

    print(" The number of chains", M, " and the array size is ", array_size)
    data_set = np.ones([M,nwalkers,int(2*run_length/thinning), ndim])  # dimension  runs    * walkers*runlength/thinning *   #parameters
    print(" The shape of the data-cube is ", data_set.shape, "(#chains,#walkers, #run_length/thinning, #dimensions)")

    # take start time
    print("*******************************************************************************************************")
    start_time = datetime.now()
    for m in range(0,M):

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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4,args=[alpha])

	    # steps first half
        sampler.run_mcmc(p0, total_length, thin=thinning)
        print(" There are ", len(p0) , "walkers to be intialized. These shape is: ",sampler.chain.shape )

	    #sampler.run_mcmc(pos, run_length, rstate0=state, thin=thinning)
	    # Print out the mean acceptance fraction. In general, acceptance_fraction
	    # has an entry for each walker so, in this case, it is a 250-dimensional
	    # vector.

        print(" Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    	# obtain the dimension of the chain
	     #print(sampler.chain.shape )    # gives (250, 1000, 50)  gives 250 vectors of 50 elements for a 'time' durations 1000 elements

        #    print("The shape of the chain obtain from sampler is: ", sampler.chain.shape)
        print(" Dimensions/parameters: ", sampler.chain.shape[2])
        print(" Walkers:               ", sampler.chain.shape[0])
        print(" Iterations:            ", sampler.chain.shape[1])

        print(" The length of parameters of walker of one chain is", len(sampler.chain[0,:,0]) )
        print("Size of the chain given by the emcee: ", sampler.chain.shape)     # 80 x 15 x 40   ( L,2*run_length, ndim)
        print("Size of the data-cube: ", data_set.shape)                         # 80 x 15 x 40   ( L,run_length, ndim)

        data_set[m,:,:,:] =  sampler.chain[:,:,:]
        #for t in range(0, int(total_length/thinning)):
        # data_set[m,:,t,:] =  sampler.chain[:,t,:]
    filename = "Gaussian_data_n="+ str(ndim)+"_j="+str(m) +"newest.txt"
    completeName = os.path.join(save_path, filename)
    write(data_set, completeName)

    
    # load data to control
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
    print("*******************************************************************************************************")

