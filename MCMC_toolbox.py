#!/usr/bin/env python
"""
    This file is toolbox for diagnostics on MCMC method.
    So far only got the Gelman-Rubin-Diagnostics
    """
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import math

def calculate_W(y):
    input_y = copy.deepcopy(y)
    s = np.shape(input_y)
    m = s[0]
    n = s[1]
    ndim = s[2]
    psi_bar = copy.deepcopy(y)
    # calculate the mean over the itterations.
    slice = np.mean(input_y, axis=1)
    
    # substract the mean from each slice
    # this is not done with linear algebra, because I didn't want to make a 10.000 by 10.000 matrix
    for i in range(0, s[1]):
        psi_bar[:,i,:] -= slice[:,:]
    x = np.tensordot(psi_bar,np.transpose(psi_bar), axes=([1,0],[1,2]))
    return x/((n-1)*m)

def calculate_B(y):
    input_y = copy.deepcopy(y)
    s = np.shape(input_y )
    m = s[0]
    n = s[1]
    ndim = s[2]
    
    # calculate the mean over the itterations.
    x = np.mean(input_y , axis=1)                 #(4,10)
    # calculate deviation scores:      a = X - 11'X(1/n)
    one = np.matrix(np.ones(m*m).reshape(m,m))    #(4,4)
    a =  x - one*x/m                              #(4,10)
    
    # caclutate b
    b = np.transpose(a).dot(a)
    return b*n/(m-1)


" This is a function that applies the multivariate Gelman-Rubin diagnostic. \
  It expects as input a three-dimensional array.\
    First dimension are the chains m, \
    Second dimension are the itterations, n \
    Thrid dimension are the parameters, ndim \
  The code is based on the Gelman-Rubin diagnostics in the CODA software package in R."
def gelman_rubin(chain):
    #print("==========================================================================")
    print("Calculating Gelman-Rubin Diagnostic, version 1.1")
    print("The shape of the input chain is", np.shape(chain))
    ndim = chain.shape[2]
    nitt= chain.shape[1]
    nchains = chain.shape[0]

    W = calculate_W(chain)
    B = calculate_B(chain)
    #find largest eigenvalue
    CW = np.linalg.cholesky(W)
    x = np.linalg.solve(CW, B)
    x2 = np.linalg.solve(CW, np.transpose(x))
    eigen =np.linalg.eigvals(x2)
    mpsrf = np.sqrt((1 - 1/nitt) + (1 + 1/ndim) * eigen[0]/nitt)
    print("mpsrf", mpsrf.real)
    return mpsrf.real


" Almost identical code compared to LaPlacesDemon in R, however slightly different due to different indexing and slicing in python. "
def IAT(x):
    #print("==========================================================================")
    #print("Calculating the Integrated Autocorrelation Time, version 1.1")
    dt = copy.deepcopy(x)
    n = int(np.shape(x)[0])
    mu = np.mean(x)
    s2 = np.cov(x)
    maxlag = int(max(3,np.floor(n/2)))
    Ga = np.zeros(2)
    Ga[0] = s2
    lg  = 0    # lg-1
    zx = dt[0:(n - lg-1)] - mu*np.ones(n - lg-1)
    zy = dt[(lg + 1):n] -mu*np.ones(n - lg-1)
    # print(zy)
    
    Ga[0] = Ga[0] + np.sum(zx*zy)/n
    #print("Ga", Ga)
    m =1
    lg = 2*m  -1
    zx = dt[0:(n - lg-1)] - mu*np.ones(n - lg-1)
    zy = dt[(lg + 1):n] -mu*np.ones(n - lg-1)
    Ga[1] =  np.sum(zx*zy)/n
    lg = 2*m + 1 -1
    #print(np.shape(zy))

    zx = dt[0:(n - lg-1)] - mu*np.ones(n - lg-1)
    zy = dt[(lg + 1):n] -mu*np.ones(n - lg-1)
    Ga[1] =  Ga[1] + np.sum(zx*zy)/n
    
    #Ga[1] <-  np.sum((dt[1:(n - lg)] - mu) * (dt[(lg + 1):n] -mu))/n
    IAT = Ga[0] / s2
    #print("test IAT",IAT)
    while ((Ga[1] > 0) & (Ga[1] < Ga[0])):
       m = m+1
       if (2 * m + 1 > maxlag):
          print("Not enough data, maxlag=", maxlag, "\n")
          break
       Ga[0] = Ga[1]
       lg = 2 * m -1
       zx = dt[0:(n - lg-1)] - mu*np.ones(n - lg-1)
       zy = dt[(lg + 1):n] -mu*np.ones(n - lg-1)
       Ga[1] = np.sum(zx*zy)/n
       lg = 2 * m+1 -1
       zx = dt[0:(n - lg-1)] - mu*np.ones(n - lg-1)
       zy = dt[(lg + 1):n] -mu*np.ones(n - lg-1)
       Ga[1] = Ga[1] +  np.sum(zx*zy)/n
       IAT = IAT + Ga[0]/s2
    #  print(IAT)
    IAT = -1 + 2*IAT
    # print("IAT:", IAT)
    return IAT
