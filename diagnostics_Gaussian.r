
# this code is for Convergence-Diagnostic on the data. 
# It uses data converted in approriate format by python program Gaussian_Load_and_Convert.py
library(coda)
library(LaplacesDemon)

# python writes in the opposite order of R. 
# 3 therefore an 3d array with dimension  in python of [ndim, runlength, m]
# becomes in R an array of [m, runlength, ndim]
# CAUTION instead of  dimensions:  [#chains, length, parameters] the dimensions is   # [length,#chains, parameters]
setwd('/Users/davidhuijser/Documents/emcee/Autocorrelation')
#means = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Gaussian_data_means_n=100_j=3NEWEST.txt', header=F)
#vars = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Gaussian_data_means_n=100_j=3NEWEST.txt', header=F)

means = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Rosenbrock_data_means_n=50_j=3NEWEST.txt', header=F)
#vars = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Rosenbrock_data_means_n=50_j=3NEWEST.txt', header=F)


vorm = dim(means)
Z_means = array( as.matrix(means),dim=c(vorm[1]/4,4, vorm[2]))
print("Print means X")
print(mean(Z_means[1,1:4,1]))
print("Print means Y")
print(mean(Z_means[1:9,1,1]))
print("Print means Z")
print(mean(Z_means[1,1,1:9]))


combinedchains = mcmc.list(as.mcmc(Z_means[,1,]), as.mcmc(Z_means[,2,]), as.mcmc(Z_means[,3,]),as.mcmc(Z_means[,4,]) ) 
#plot(combinedchains)
(gelman.diag)(combinedchains, multivariate=TRUE, autoburnin=FALSE,transform = FALSE)
#gelman.plot(combinedchains)
heidel.diag(as.mcmc(Z_means[,1,]), eps=1.0, pvalue=0.05)
heidel.diag(as.mcmc(Z_means[,2,]), eps=1.0, pvalue=0.05)
heidel.diag(as.mcmc(Z_means[,3,]), eps=1.0, pvalue=0.05)
heidel.diag(as.mcmc(Z_means[,4,]), eps=1.0, pvalue=0.05)


# test 
IAT(Z_means[,1,])
IAT(Z_means[,2,])
IAT(Z_means[,3,])
IAT(Z_means[,4,])
