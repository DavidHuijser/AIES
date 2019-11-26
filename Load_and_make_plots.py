#!/usr/bin/env python
"""
code for loading and plotting correlated for thesis.
"""


from __future__ import print_function
from numpy import loadtxt
import numpy as np
import os.path
from datetime import datetime

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import copy
import pylab
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import emcee

from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
plt.ion()


import pickle
import os.path

def read(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data

load_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
#save_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation'
img_path = '/Users/davidhuijser/Documents/emcee/Autocorrelation/images'


start_time = datetime.now()


def make_table(x,ndim_counter,ndim, basename):
  filename = basename + "_var_and_mean_results_averages.txt"
  completeName = os.path.join(save_path, filename)
  target = open(filename, 'w')
  s = np.shape(x)
  length = len(x)
  print(np.shape(x))
  mean  = np.mean(x[3,:,int(0.5*length):-1])
  var   = np.var(x[3,:,int(0.5*length):-1])
  print(mean,"+/- ", var)
  # calculate the average over all the walkers over the last 50%
  #mean = np.mean(x[)
#var =

def make_empirical_plot(x,ndim_counter,ndim, basename,length, thinning):
    # datacube, nwalkers,run_length, thinning
    emperical_mean    =  np.mean(x,axis=0)
    emperical_variance = np.var(x, axis=0)

    #from matplotlib import rc
    #import matplotlib.font_manager
    #matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    from matplotlib import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    #rc('font',**{'family':'Times New Roman','Times New Roman':['Times New Roman']})
    #rc('text', usetex=True)

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#   rc('font',**{'family':'serif','serif':['Palatino']})
#plt.rcParams['font.family'] = 'serif'
#rc('text', usetex=True)
    
    
    #xval =  100.00*np.arange(len(x[:,:].flatten()))/(len(x[:,:].flatten()))
    # number of likelihood evaluations: nwalkers*length*thininng
    # percentage
    L_eval =  np.arange(nwalkers,length*nwalkers+nwalkers,nwalkers, dtype=float)
    L_eval = 100*L_eval/max(L_eval)
    print(L_eval.shape,L_eval)
    # create out put
    print( " Create Flat emperical trace plots  \n")
    #matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif']
    fig = plt.figure(figsize=(8,6), dpi=300)
    plt.subplots_adjust(left=0.20, right=0.92, top=0.97, bottom=0.13)
    ax = fig.add_subplot(111)
    ax = plt.gca()
    if ndim_counter < 3:
        true_mean = np.zeros(length)
        true_sd = np.ones(length)
    else:
        true_mean = np.ones(length)
        true_sd = np.ones(length)*0.7

            #plt.plot(L_eval,x[:,:].flatten(),color='grey')

    #plt.plot(L_eval,true_mean ,ls='dashed', color='lightgrey',linewidth=1,label=r'$true$ $mean$')
    #plt.plot(L_eval,true_sd, ls='dashdot', color='lightgrey',linewidth=1, label=r'$true$ $SD$')
    if ndim_counter < 3:
        plt.axis([min(L_eval),max(L_eval), -1.0, 2.5])
    else:
        plt.axis([min(L_eval),max(L_eval), -1.0, 2.5])
    plt.plot(L_eval,emperical_mean , color='grey', label=r'$emperical$ $mean$',ls='solid',linewidth=1)
    plt.plot(L_eval,emperical_variance, color='grey', label=r'$emperical$ $SD$',ls='dotted',linewidth=1)

    plt.plot(L_eval,true_mean ,ls='dashed', color='black',linewidth=0.5,label=r'$true$ $mean$')
    plt.plot(L_eval,true_sd, ls='dashdot', color='black',linewidth=0.5, label=r'$true$ $SD$')

    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$progress$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    plt.legend(loc=2, borderaxespad=0., labelspacing=0.09,)
#fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
#   xticks = ticker.FormatStrFormatter(fmt)
#   ax.xaxis.set_major_formatter(xticks)
    plt.gca().set_xticklabels(['{:.0f}\%'.format(x) for x in plt.gca().get_xticks()])



 #   number ='%.4f' % np.std(new_set[:,:,0].flatten())
 #   textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
 #   props = dict(boxstyle='round', facecolor='white', alpha=1.0)
 #   ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

  # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#   ax.xaxis.major.formatter._useMathText = True

    plt.show()
    filename = basename + '_emperical_traceplot_N_'+ str(ndim) +'_newest.png'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)

    filename = basename + '_emperical_traceplot_N_'+ str(ndim) +'_newest.eps'
    filename = os.path.join(img_path, filename)

#   plt.savefig(filename)
    plt.close(fig)





def make_hexbin_plot(x,y, ndim_counter,ndim, basename):
    print(" Create Hexagon bin plot")
    from matplotlib import rc
    import matplotlib.font_manager
    matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

#plt.rcParams["font.family"] = "cursive"
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
    rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rcParams['font.family'] = 'serif'
    #     fig2 = plt.figure(figsize=(8,6), dpi=300)
    # ax = fig2.add_subplot(111)
    #plt.subplots_adjust(left=0.14, right=0.95, top=0.96, bottom=0.13)
    
    #fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax= fig.add_subplot(111)
    fig.subplots_adjust(hspace=1.0, left=0.15, right=0.95, top=0.96, bottom=0.13)
    # ax = axs[0]
    hb = ax.hexbin(x, y, gridsize=30, cmap='gray_r')
    #ax.set_title("Hexagon binning")
    if ndim_counter <3:
        ax.axis([-5, 5, -5, 5])
    else:
        ax.axis([-10, 10, -10, 10])
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')

    plt.xlabel(r'$y_1$',fontsize=size_of_font)
    plt.ylabel(r'$y_2$',fontsize=size_of_font)

    plt.tick_params(axis='both', which='major', labelsize=size_of_font)
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)

# plt.set_cmap('gray_r')

    filename = basename + '_hexbin_plot_N_'+ str(ndim) +'_newest.png'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)

    filename = basename + '_hexbin_plot_N_'+ str(ndim) +'_newest.eps'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)
    plt.close(fig)


def obtain_running_average(x):
    start_time = datetime.now()
    length = len(x)
    running_average  =   np.zeros(len(x))
    running_average[0] = x[0]
    running_average[1] = np.mean(x[0:2])
    for j in range(2,len(x)):
        counter = int(round(j/2))
        #print("j=",j , " j-counter= ",j-counter,":",j+1, " ", x[j-counter:j+1]," ", np.mean(x[j-counter:j+1]))
        running_average[j] = np.mean(x[j-counter:j+1])
    end_time = datetime.now()
    print('Duration average: {}'.format(end_time - start_time))
    return(running_average)

def obtain_running_variance(x):
    start_time = datetime.now()
    length = len(x)
    running_variance  =  np.zeros(len(x))
    running_variance[0] =  0
    running_variance[1] = np.abs(x[1] - np.mean(x[0:1]))
    for j in xrange(2,len(x)):
        counter = int(round(0.5*j))
        mean = x[j-counter:j+1]  - np.mean(x[j-counter:j+1])
        running_variance[j] = np.sqrt(np.mean(mean*mean))
        #mean_val  = np.mean(x[j-counter:j+1])
        #running_variance[j] = np.mean(np.abs(x[j-counter:j+1]  - mean_val))
    end_time = datetime.now()
    print('Duration variance quick: {}'.format(end_time - start_time))
    return(running_variance)






def make_traceplot(x, ndim_counter,ndim, basename):
    print(np.shape(x))
    y = x[:,:].flatten()
    
    running_average_of_last_fifty_percent = obtain_running_average(y)
    running_variance_of_last_fifty_percent = obtain_running_variance(y)

    
    from matplotlib import rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    
    fig2 = plt.figure(figsize=(8,6), dpi=300)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.20, right=0.92, top=0.97, bottom=0.13)
    ax = plt.gca()

    xval =  100.00*np.arange(len(y), dtype=float)/(len(y))

    length = len(y)
    plt.plot(xval,y,color='grey')

    if ndim_counter < 3:
        true_mean = np.zeros(length)
        true_sd = np.ones(length)
        plt.axis([0,100, -3.5, 3.5])

    else:
        true_mean = np.ones(length)
        true_sd = np.ones(length)*0.7
        plt.axis([0,100, -2.5, 3.0])

#   plt.plot(xval,running_average_of_last_fifty_percent1, color='red', label=r'$running$ $old$',ls='--',linewidth=2)
# plt.plot(xval,running_average_of_last_fifty_percent2,  color='blue', label=r'$running$ $new$',ls=':',linewidth=2)

#   plt.plot(xval,running_variance_of_last_fifty_percent1, color='red', label=r'$running$ $old$',ls='--',linewidth=2)
#   plt.plot(xval,running_variance_of_last_fifty_percent2,  color='blue', label=r'$running$ $new$',ls=':',linewidth=2)


    plt.plot(xval,true_mean ,ls='--', color='grey',linewidth=1)
    plt.plot(xval,true_sd, ls=':', color='grey',linewidth=1)
	#[ '-' | '--' | '-.' | ':' | 'steps' | ...]
    plt.plot(xval,running_average_of_last_fifty_percent , color='black', label=r'$running$ $mean$',ls='--',linewidth=2)
    plt.plot(xval,running_variance_of_last_fifty_percent, color='black', label=r'$running$ $SD$',ls=':',linewidth=2)
    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$progress$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)

    plt.legend(loc=4, borderaxespad=0., labelspacing=0.09,)
    plt.gca().set_xticklabels(['{:.0f}\%'.format(x) for x in plt.gca().get_xticks()])

    filename = basename + '_flat_traceplot_N_'+ str(ndim) +'_newest.png'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)
    
    filename = basename + '_flat_traceplot_N_'+ str(ndim) +'_newest.eps'
    filename = os.path.join(img_path, filename)
    
    plt.savefig(filename)
    plt.close(fig2)


def make_density_plot(x,y, ndim_counter,ndim, basename):
    print(" Create 2D scatter plot")
    fig2 = plt.figure(figsize=(8,6), dpi=300)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.14, right=0.95, top=0.96, bottom=0.13)
    if ndim_counter <3:
        xedges, yedges = np.linspace(-5, 5, 42), np.linspace(-5, 5, 42)
    else:
        xedges, yedges = np.linspace(-10, 10, 32), np.linspace(-10, 10, 32)
       # xedges, yedges = np.linspace(-10, 10, 42), np.linspace(-10, 10, 42)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]

    plt.scatter(x, y, c=c, alpha=0.1)
    if ndim_counter <3:
        plt.axis([-5, 5, -5, 5])
    else:
        #plt.axis([-10, 10, -10, 10])
        plt.axis([-10, 10, -10, 10])
    plt.xlabel(r'$y_1$',fontsize=size_of_font)
    plt.ylabel(r'$y_2$',fontsize=size_of_font)

    plt.tick_params(axis='both', which='major', labelsize=size_of_font)
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
#    plt.colorbar(fig2, format='%.0e')
#    plt.set_cmap('winter')

#    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,  norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
# cbar.set_clim(10.0, 200.0)
    cbar.ax.set_ylabel('Density')
    cbar.ax.yaxis.set_offset_position('right')
    cbar.update_ticks()
    plt.set_cmap('gray_r')

    filename = basename + '_histogram_plot_N_'+ str(ndim) +'_newest.png'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)
    
    filename = basename + '_histogram_plot_N_'+ str(ndim) +'_newest.eps'
    filename = os.path.join(img_path, filename)
    plt.savefig(filename)
    plt.show()
    plt.close(fig2)


# take start time
start_time = datetime.now()

size_of_font = 20
size_of_label = 16

#from matplotlib import rc
#import matplotlib.font_manager
#matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

#plt.rcParams["font.family"] = "cursive"
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

# take start time
start_time = datetime.now()
# Info on the orginal run (universal for all runs and problems)
ndims =     [10, 50 , 100]
#ndims = [10]
M = 4   # amount of chains for the Gelman & Rubin analysis
run_length  = int(1e5)
#half_length =   run_length
total_length = 2*run_length

# steps = runlength
# dimension of data cube  (walkers  , length/thinning, ndim )   example:     #(250, 1000, 50)    = (#walkers, #steps, # parameters)
#200 x 200 X 10  =  #walkers * #steps * #dim

# the initial distribution should be overly-dispersed
print("******************************************************************************")
print("This is the newest analysis of the emcee producing all plots  ")
print(" ")
for ndim_counter in range(3,6):
    ndim = ndims[ndim_counter % 3]
    print("\n The number of parameters: ",ndim)
    if ndim_counter <3:
        filename = "Gaussian_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST.txt"
        basename = 'Gaussian'
    else:
        filename = "Rosenbrock_data_n="+ str(ndim)+"_j="+str(M-1) +"NEWEST.txt"
        basename = 'Rosenbrock'

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
    length = int(loaded_data_set.shape[2])


    # density plot over 2nd half, 1def and 2ghi
    x = loaded_data_set[3,:,int(0.5*length):int(length),0].flatten()
    y = loaded_data_set[3,:,int(0.5*length):int(length),1].flatten()
    #make_density_plot(x,y, ndim_counter,ndim, basename)

    #make_table(loaded_data_set[:,:,:,0], ndim_counter,ndim, basename)

    make_traceplot(loaded_data_set[3,:,:,0], ndim_counter,ndim, basename)

    make_empirical_plot(loaded_data_set[3,:,:,0], ndim_counter,ndim, basename, length, thinning)

    #STOP
    
    #emperical_mean     = obtain_emperical_mean(loaded_data_set[3,:,:,0],nwalkers,2*run_length,thinning)
    #emperical_variance = obtain_emperical_variance(loaded_data_set[3,:,:,0],nwalkers,2*run_length,thinning)
    
    # running averages
    #running_average_of_last_fifty_percent = obtain_running_average(loaded_data_set[3,:,:,0].flatten() )
    #running_variance_of_last_fifty_percent = obtain_running_variance(loaded_data_set[3,:,:,0].flatten() )

#all_mean[ndim_counter, m+1] = running_average_of_last_fifty_percent[-1]
#all_var[ndim_counter, m+1]  = np.sqrt(running_variance_of_last_fifty_percent[-1])


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("*******************************************************************************************************")

STOP

# steps = runlength 
run_length  = 5000
half_length =   run_length
total_length = 2*run_length

# steps = runlength 
# dimension of data cube  (walkers  , length/thinning, ndim )   example:     #(250, 1000, 50)    = (#walkers, #steps, # parameters)
                        #200 x 200 X 10  =  #walkers * #steps * #dim           

# the initial distribution should be overly-dispersed

print("******************************************************************************")
print("This is the original analysis of the emcee using a correlated Gaussian ")
print("Loading the data")

all_mean = np.zeros([len(ndims),( M+1)])
all_var = np.zeros([len(ndims),(M+1)])
for ndim_counter in range(0,len(ndims)):
    ndim = ndims[ndim_counter]
    thinning =  thinnings[ndim_counter]
    print("\n The number of parameters: ",ndim)
    nwalkers = 2*ndim
    print(" The number of walkers: ",nwalkers)
    print(" The thinning is: ",thinning)
    print(" The runlength is: ",run_length," steps")
    print(" The chain-length is: ",run_length,"/",thinning,'=',run_length/thinning)

    # the size of the array for storage is  run_length*2*ndinm*ndim/thinning
    array_size = 2*M*nwalkers*ndim*run_length/thinning

    print(" The number of chains", M, " and the array size is ", array_size)
    new_set = np.ones([nwalkers, 2*half_length/thinning, ndim])  # dimension  runs    * walkers*runlength/thinning *   #parameters
    print(" The shape of the data-cube is ", new_set.shape, "(#chains,#walkers, #run_length/thinning, #dimensions) \n")
    print(" For this analysis we DONT discard the burn in")

    #zvalues_set = np.ones((M,nwalkers,2*ndim/thinning))
    #zvalues_subset= np.ones((M,2*ndim/thinning))

    all_mean[ndim_counter, 0] = ndim
    all_var[ndim_counter, 0]  = ndim

    for m in range(0,M):
        m=3
        filename = "Gaussian_data_n="+ str(ndim)+"_j="+str(m) +".txt"
        completeName = os.path.join(save_path, filename)         
        loaded_data_set = read(completeName)

        print(" Compare between two sets. First walker", np.mean(loaded_data_set[m,1,:,:]) )
        print(" Compare between two sets. Last-step", np.mean(loaded_data_set[m,:,-1,:]))
        print(" Compare between two sets. First parameter-step", np.mean(loaded_data_set[m,:,:,0]))
     

        print("  \n Shapes of the Array \n")
        print("One chain:" , new_set.shape, "   \n ")
        print("All chains", loaded_data_set.shape, "   \n ")

        for t in range(0, total_length/thinning):
#                data_set[m,:,t,:] =  sampler.chain[:,t,:]
                 new_set[:,t,:] = loaded_data_set[m,:,t,:] 
   
      	length = len(new_set[:,:,0].flatten() )                      # obtain length
	true_mean = np.zeros(length)
	true_sd = np.zeros(length)+1.0


	emperical_mean     = obtain_emperical_mean(new_set[:,:,0],nwalkers,2*run_length,thinning)
	emperical_variance = obtain_emperical_variance(new_set[:,:,0],nwalkers,2*run_length,thinning)

	running_average_of_last_fifty_percent = obtain_running_average(new_set[:,:,0].flatten() )
	running_variance_of_last_fifty_percent = obtain_running_variance(new_set[:,:,0].flatten() )
   

        all_mean[ndim_counter, m+1] = running_average_of_last_fifty_percent[-1]
        all_var[ndim_counter, m+1]  = np.sqrt(running_variance_of_last_fifty_percent[-1])



	# density plot
    x = new_set[:,:,0].flatten()
    y = new_set[:,:,1].flatten()
    print(" Create 2D scatter plot")
    fig2 = plt.figure(figsize=(8,6), dpi=300)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)

    xedges, yedges = np.linspace(-10, 10, 42), np.linspace(-10, 10, 42)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]
    plt.scatter(x, y, c=c, alpha=0.1)

#    plt.hist2d(x, y, bins=40, norm=LogNorm())
#    plt.scatter(x,y,edgecolors='none',s=marker_size,c=0.2, norm=matplotlib.colors.LogNorm())
    plt.axis([-10, 10, -10, 10])
    plt.xlabel(r'$x_1$',fontsize=size_of_font)
    plt.ylabel(r'$x_2$',fontsize=size_of_font)

    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)  

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
#    plt.colorbar(fig2, format='%.0e')
#    plt.set_cmap('winter')

#    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,  norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
    cbar.set_clim(10.0, 200.0)
    cbar.ax.set_ylabel('Density')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('right')                         
    cbar.update_ticks()
    plt.set_cmap('gray_r')

#    plt.colorbar(fig2, format='%.0e')
#    plt.colorbar(fig2, format=ticker.FuncFormatter(fmt))

    filename = 'Gaussian_histogram_plot_N_'+ str(ndim) +'_new.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)

    filename = 'Gaussian_histogram_plot_N_'+ str(ndim) +'_new.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig2)


      # create out put
    print( " Create Flat emperical trace plot \n")
    length = len(new_set[0,:,0].flatten() )                      # obtain length
    true_mean = np.zeros(length)
    true_sd = np.zeros(length)+1.0

    fig = plt.figure(figsize=(8,6), dpi=300)
    plt.subplots_adjust(left=0.145, right=0.95, top=0.96, bottom=0.125)
    ax = fig.add_subplot(111)    
    plt.plot(true_mean ,ls=':')
    plt.plot(true_sd, ls=':')

    plt.plot(emperical_mean , color='grey', label='emprical mean',ls='--',linewidth=2)
    plt.plot(np.sqrt(emperical_variance), color='black', label='empircal SD',ls='-.',linewidth=2)

    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$iteration$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)
    plt.legend(loc=4, borderaxespad=0.)
   # number ='%.4f' % np.std(new_set[:,:,0].flatten())
   # textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
   # props = dict(boxstyle='round', facecolor='white', alpha=1.0)
#    ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True
    plt.axis([0,length, -0.5, 2.0])       
 #   s = str(alpha) 
#    s = s.replace('.','')
    plt.gray()

    filename = 'Gaussian_flat_emperical_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)

    filename = 'Gaussian_flat_emperical_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.close(fig)


    # create out put
    print( " Create Flat emperical trace plot in color  \n")
    fig = plt.figure(figsize=(8,6), dpi=300)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.11)
    ax = fig.add_subplot(111)    

    plt.plot(true_mean ,ls=':')
    plt.plot(true_sd, ls=':')

    plt.plot(emperical_mean , color='red', label='emprical mean',ls='--',linewidth=2)
    plt.plot(np.sqrt(emperical_variance), color='black', label='empircal SD',ls='-.',linewidth=2)

    plt.ylabel(r'$x_1$',fontsize=16)
    plt.xlabel(r'$iteration$',fontsize=16)
    plt.legend(loc=4, borderaxespad=0.)
 #   number ='%.4f' % np.std(new_set[:,:,0].flatten())
 #   textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
 #   props = dict(boxstyle='round', facecolor='white', alpha=1.0)
 #   ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True
    plt.axis([0,length, -0.5, 2.0])       
#    s = str(alpha) 
#    s = s.replace('.','')
    filename = 'Gaussian_color_flat_emperical_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)

    filename = 'Gaussian_color_flat_emperical_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    
    plt.close(fig)


	# density plot
    x = new_set[:,:,0].flatten()
    y = new_set[:,:,1].flatten()

	# Plot data
#	fig1 = plt.figure()
    fig1 = plt.figure(figsize=(8,6), dpi=300)
    ax = fig1.add_subplot(111)
    fig1.subplots_adjust(top=0.85)
    plt.plot(x,y,'.r')
    plt.close(fig1)

	#ax.set_title('axes title')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
	# Estimate the 2D histogram
    nbins = 100
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    print("total is", sum(sum(H)), "and", nwalkers*run_length)
	# H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
 
	# Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zer


    # Plot 2D histogram using in grey
    print( " Create 2D grey histogram")
    fig2 = plt.figure(figsize=(8,6), dpi=300)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)
    plt.pcolormesh(xedges,yedges,Hmasked/(nwalkers*run_length),vmin=0.0, vmax=0.005)
    plt.axis([-5, 5, -2, 8])
    plt.xlabel(r'$x_1$',fontsize=size_of_font)
    plt.ylabel(r'$x_2$',fontsize=size_of_font)
    cbar = plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)  
    cbar.ax.set_ylabel('Density')
   
#    plt.colorbar(fig2, format='%.0e')
#    plt.colorbar(fig2, format=ticker.FuncFormatter(fmt))

    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('right')                         
    cbar.set_clim(0.0, 100.0)
    cbar.update_ticks()
    plt.set_cmap('gray_r')
#    plt.set_cmap('winter')
#    s = str(alpha) 
#    s = s.replace('.','')
    filename = 'Gaussian_counts_plot_N_grey_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)

    filename = 'Gaussian_counts_plot_N_grey_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig2)


	# Plot 2D histogram using pcolor
    print( " Create 2D color histogram")
#	fig2 = plt.figure()
    fig2 = plt.figure(figsize=(8,6), dpi=300)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)
    ax = fig2.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hmasked/(nwalkers*run_length),vmin=0.0, vmax=0.005)    
    plt.axis([-5, 5, -2, 8])
    plt.xlabel(r'$x_1$',fontsize=size_of_font)
    plt.ylabel(r'$x_2$',fontsize=size_of_font)
    cbar = plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)
    cbar.ax.set_ylabel('Density')

    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('right')                         
    cbar.update_ticks()

    plt.set_cmap('jet')
   # s = str(alpha) 
#    s = s.replace('.','')
    filename = 'Gaussian_color_counts_plot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    filename = 'Gaussian_color_counts_plot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig2)


    print(" Making normal trace plots COLOR ")
 	#print("the length of parameters of walker of one chain is", len(sampler.lnprob) ) 
    sd = np.std(new_set[:,:,0].flatten()) 
    fig = plt.figure(figsize=(8,6), dpi=300)
	#fig.suptitle(r'Flattend traceplot of parameter  $x_0$ for all walkers ', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)    
    plt.subplots_adjust(left=0.145, right=0.96, top=0.97, bottom=0.13)
    plt.plot(new_set[0,:,0], label='$j=1$',color=('0.0'))
#	plt.plot(new_set[1,:,0], label='$j=2$',color=('0.1'),ls=':') 
    plt.plot(new_set[2,:,0],label='$j=3$',color=('0.4'))
#	plt.plot(new_set[3,:,0] ,label='$j=4$',color=('0.5'),ls=':')
    plt.plot(new_set[3,:,0], label='$j=5$',color=('0.8'))

#        plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.08)
#	plt.plot(new_set[0,:,0],'r', label='$j=1$',color=('0.1'),ls='-.')
#	plt.plot(new_set[1,:,0],'b', label='$j=2$',color=('0.2'),ls='--')
#	plt.plot(new_set[2,:,0],'g', label='$j=3$',color=('0.3'),ls=':')
#	plt.plot(new_set[3,:,0],'k', label='$j=4$',color=('0.4'))
#	plt.plot(new_set[4,:,0],label='$j=5$',color='gray')
    plt.plot(true_mean ,ls=':',color=('0.5'))
    plt.plot(true_sd, ls='-.',color=('0.5'))
    plt.ylabel(r'$x_i^j$',fontsize=size_of_font)
    plt.xlabel(r'$t$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)  

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True

    plt.legend(loc=4, borderaxespad=0., labelspacing=0.09,borderpad=0.01)
#        plt.legend(bbox_to_anchor=(0.833, 1), loc=2, borderaxespad=0., labelspacing=0.09,borderpad=0.01)
    plt.axis([0, len(new_set[0,:,0]), -3.5, 3.5])

#    s = str(alpha) 
#    s = s.replace('.','')
    filename = 'Gaussian_normal_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    filename = 'Gaussian_normal_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig)

    print("   Making normal trace plots COLOR ")
#print("the length of parameters of walker of one chain is", len(sampler.lnprob) ) 
    sd = np.std(new_set[:,:,0].flatten()) 
    fig = plt.figure(figsize=(8,6), dpi=300)
	#fig.suptitle(r'Flattend traceplot of parameter  $x_0$ for all walkers ', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.14, right=0.96, top=0.97, bottom=0.13)
#        plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.08)
    plt.plot(new_set[0,:,0],'r', label='$j=1$')
    plt.plot(new_set[1,:,0],'b', label='$j=2$')
    plt.plot(new_set[2,:,0],'g', label='$j=3$')
    plt.plot(new_set[3,:,0],'k', label='$j=4$')
#    plt.plot(new_set[4,:,0],'y',label='$j=5$')
    plt.plot(true_mean ,ls=':',color=('0.5'))
    plt.plot(true_sd, ls='-.',color=('0.5'))

    plt.ylabel(r'$x_i^j$',fontsize=size_of_font)
    plt.xlabel(r'$t$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)  

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True

    plt.legend(loc=4, borderaxespad=0., labelspacing=0.09,borderpad=0.01)
#        plt.legend(bbox_to_anchor=(0.833, 1), loc=2, borderaxespad=0., labelspacing=0.09,borderpad=0.01)
    plt.axis([0, len(new_set[0,:,0]), -3.5, 3.5])

 #   s = str(alpha) 
  #  s = s.replace('.','')
    filename = 'Gaussian_color_normal_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    filename = 'Gaussian_color_normal_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig)

    print("   Making flat trace plots GREY ")
    sd = np.std(new_set[:,:,0].flatten()) 
    fig = plt.figure(figsize=(8,6), dpi=300)
    #fig.suptitle(r'Flattend traceplot of parameter  $x_0$ for all walkers ', fontsize=14, fontweight='bold') 
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    print(new_set[:,:,0].shape)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.13)
    plt.plot(new_set[:,:,0].flatten(),color='grey')
    plt.plot(true_mean ,ls='--', color='grey',linewidth=1)
    plt.plot(true_sd, ls=':', color='grey',linewidth=1)
	#[ '-' | '--' | '-.' | ':' | 'steps' | ...]
    plt.plot(running_average_of_last_fifty_percent , color='black', label=r'$running$ $mean$',ls='--',linewidth=2)
    plt.plot(np.sqrt(running_variance_of_last_fifty_percent), color='black', label=r'$running$ $SD$',ls=':',linewidth=2)
    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$iteration$',fontsize=size_of_font)
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font) 
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label) 
    plt.legend(loc=4, borderaxespad=0., labelspacing=0.09,)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True
#	number ='%.4f' % np.std(new_set[:,:,0].flatten())
#	sigma = np.std(new_set[:,:,0].flatten())  
    plt.axis([0,len(new_set[:,:,0].flatten()) , -4.0, 4.0])
 #       textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,running_average_of_last_fifty_percent[-1], np.sqrt(running_variance_of_last_fifty_percent[-1]))
#	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
#	ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
  #  s = str(alpha) 
   # s = s.replace('.','')
    filename = 'Gaussian_flat_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    filename = 'Gaussian_flat_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show()
    plt.close(fig)  

    print("   Making flat trace plots COLOR ")
    fig = plt.figure(figsize=(8,6), dpi=300)
    #fig.suptitle(r'Flattend traceplot of parameter  $x_0$ for all walkers ', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    print(new_set[:,:,0].shape)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.13)
    plt.plot(new_set[:,:,0].flatten(),color='blue',linewidth=1)
    plt.plot(true_mean ,ls='--', color='black')
    plt.plot(true_sd, ls=':', color='black')
	#[ '-' | '--' | '-.' | ':' | 'steps' | ...]

    plt.plot(running_average_of_last_fifty_percent , color='red', label=r'$running$ $mean$',ls='--',linewidth=2)
    plt.plot(np.sqrt(running_variance_of_last_fifty_percent), color='green', label=r'$running$ $SD$',ls=':',linewidth=2)
    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$iteration$',fontsize=size_of_font)
    plt.legend(loc=4, borderaxespad=0., labelspacing=0.09,)

    plt.axis([0,len(new_set[:,:,0].flatten()) , -4.0, 4.0])
    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)  

#	number ='%.4f' % np.std(new_set[:,:,0].flatten())
#	sigma = np.std(new_set[:,:,0].flatten())  
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True
	

    plt.axis([0,len(new_set[:,:,0].flatten()) , -4.0, 4.0])
 #       textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,running_average_of_last_fifty_percent[-1], np.sqrt(running_variance_of_last_fifty_percent[-1]))
#	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
#	ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
  #  s = str(alpha) 
  #  s = s.replace('.','')
    filename = 'Gaussian_color_flat_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    filename = 'Gaussian_color_flat_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)         
    plt.savefig(filename)
    plt.show() 
    plt.close(fig)


    print("-----------------------------------------------------------------------------------------")
    #stop 
    print("-----------------------------------------------------------------------------------------")
       



    # create out put  for the q_values
    print( " Create mean and variance trace plot \n")
    fig = plt.figure(figsize=(8,6), dpi=300)	
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
#        plt.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.08)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.13)
    plt.plot(emperical_mean , color='black', label=r'$empirical$ $mean$ $x_1$',ls='--',linewidth=1)
    plt.plot(np.sqrt(emperical_variance), color=r'grey', label='$empirical$ $SD$ $x_1$',ls='-.',linewidth=1)
    plt.ylabel(r'$x_1$',fontsize=size_of_font)
    plt.xlabel(r'$iteration$',fontsize=size_of_font)
    plt.legend(loc=1, borderaxespad=0.)
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
    ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.axis([0,length, -0.5, 6.0])   

    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
    ax.tick_params(axis='x', labelsize=size_of_label)
    ax.tick_params(axis='y', labelsize=size_of_label)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True

    filename = 'Gaussian_mean_and_variance_traceplot_N_'+ str(ndim) +'.png'
    filename = os.path.join(img_path, filename)   
    plt.savefig(filename)

    filename = 'Gaussian_mean_and_variance_traceplot_N_'+ str(ndim) +'.eps'
    filename = os.path.join(img_path, filename)   
    plt.savefig(filename)
    plt.close(fig)




#    # create out put  for the q_values
#    print( " Create mean and variance trace plot in COLOR \n")
#    fig = plt.figure(figsize=(8,6), dpi=300)	
#    ax = fig.add_subplot(111)
##        plt.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.08)
#    plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.11)
#    plt.plot(emperical_mean , color='blue', label=r'$empirical$ $mean$ $x_1$',ls='--',linewidth=1)
#    plt.plot(np.sqrt(emperical_variance), color='red', label='$empirical$ $SD$ $x_1$',ls='-.',linewidth=1)
#    plt.ylabel(r'$x_1$',fontsize=size_of_font)
#    plt.xlabel(r'$iteration$',fontsize=size_of_font)
#    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
#    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
#    ax.tick_params(axis='x', labelsize=size_of_label)
#    ax.tick_params(axis='y', labelsize=size_of_label)
#    plt.legend(loc=1, borderaxespad=0.)
#    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
#    ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
#    plt.axis([0,length, -0.5, 6.0])   
#    filename = 'Gaussian_color_mean_and_variance_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.png'
#    plt.savefig(filename)
#    filename = 'Gaussian_color_mean_and_variance_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.eps'
#    plt.savefig(filename)
#    plt.close(fig)

#    # create out put  for the q_values
#    # create out put  for the q_values
#    print( " Create q-emperical trace plot \n")
#    fig = plt.figure(figsize=(8,6), dpi=300)	
#    ax = fig.add_subplot(111)
#    q_set  = calculate_qs(new_set, nwalkers, run_length,ndim, alpha, thinning) 
#    print("Shape ", q_set.shape)
#    q_variance = obtain_emperical_variance_q_(q_set,nwalkers,run_length, thinning)
##        plt.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.08)
#    plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.11)
#    plt.plot(np.sqrt(q_variance), color='black')
#    plt.ylabel(r'$\hat{\sigma}_{\mathbf{q}}$',fontsize=size_of_font)
#    plt.xlabel(r'$iteration$',fontsize=size_of_font)
#    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
#    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
#    ax.tick_params(axis='x', labelsize=size_of_label)
#    ax.tick_params(axis='y', labelsize=size_of_label)
##        plt.legend(bbox_to_anchor=(0.4, 1), loc=1, borderaxespad=0.)
# #       plt.legend(loc=1, borderaxespad=0.)
##	number ='%.4f' % np.std(new_set[:,:,0].flatten())
# #       textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
##	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
##	ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
#    plt.axis([0,60, -0.5, 6.0]) 
#    plt.tick_params(axis='both', which='major', labelsize=14)  
#    plt.tick_params(axis='both', which='minor', labelsize=12)
#    ax.tick_params(axis='x', labelsize=25)


#    plt.gray()  

#    filename = 'Gaussian_q_emperical_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.png'
#    plt.savefig(filename)

#    filename = 'Gaussian_q_emperical_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.eps'
#    plt.savefig(filename)
#    plt.close(fig)

#    # create out put  for the q_values
#    print( " Create q-emperical trace plot COLOR \n")
#    fig = plt.figure(figsize=(8,6), dpi=300)	
#    ax = fig.add_subplot(111)
#    q_set  = calculate_qs(new_set ,nwalkers, run_length,ndim, alpha, thinning) 
#    print("Shape ", q_set.shape)
#    q_variance = obtain_emperical_variance_q_(q_set,nwalkers,run_length, thinning)
##        plt.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.08)
#    plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.11)
#    plt.plot(np.sqrt(q_variance), color='black')
#    plt.ylabel(r'$\hat{\sigma}_{\mathbf{q}}$',fontsize=size_of_font)
#    plt.xlabel(r'$iteration$',fontsize=size_of_font)

#    plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
#    plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
#    ax.tick_params(axis='x', labelsize=size_of_label)
#    ax.tick_params(axis='y', labelsize=size_of_label)
##        plt.legend(bbox_to_anchor=(0.4, 1), loc=1, borderaxespad=0.)
# #       plt.legend(loc=1, borderaxespad=0.)
##	number ='%.4f' % np.std(new_set[:,:,0].flatten())
# #       textstr = '$\mathrm{N}=%.2f$\n$\hat{\mu}=%.4f$\n$\hat{\sigma}=%.4f$'%(ndim,emperical_mean[-1], np.sqrt(emperical_variance[-1]))
##	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
##	ax.text(0.025, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
#    plt.axis([0,60, -0.5, 6.0]) 
#    plt.gray()  
#    filename = 'Gaussian_color_q_emperical_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.png'
#    plt.savefig(filename)

#    filename = 'Gaussian_color_q_emperical_traceplot_alpha_' + s +'_N_'+ str(ndim) +'.eps'
#    plt.savefig(filename)

#    plt.close(fig)
#    plt.close(fig)

    
#	print("mean values: ", running_average_of_last_fifty_percent[-1])
#	print("variance: ",np.sqrt(running_variance_of_last_fifty_percent[-1]))
       
filename = "Gaussian_var_and_mean_results_averages.txt"
completeName = os.path.join(save_path, filename)       
target = open(filename, 'w')
 # line1 = "mean values: " +  str(running_average_of_last_fifty_percent[-1])
 # line2 = "variance: " + str(np.sqrt(running_variance_of_last_fifty_percent[-1]))
 
for ndim_counter in range(0,len(ndims)):
#   for m in range(0,M+1):
      line1 = " " + str(all_mean[ndim_counter, 0]) + "mean:"  + str(np.mean(all_mean[ndim_counter, 1:-1]) )
      target.write(line1)
      target.write("\n")
target.write("\n")
for ndim_counter in range(0,len(ndims)):
 #  for m in range(0,M+1):
      line1 = " " + str(all_var[ndim_counter, 0]) + "mean:"  + str(np.mean(all_var[ndim_counter, 1:-1]) )
      target.write(line1)
      target.write("\n")
target.write("\n")
target.close()


filename = "Gaussian_var_and_mean_results_M="+str(m)+"_part2.txt"
completeName = os.path.join(save_path, filename)       
target = open(filename, 'w')
#  line1 = "mean values: " +  str(running_average_of_last_fifty_percent[-1])
 # line2 = "variance: " + str(np.sqrt(running_variance_of_last_fifty_percent[-1]))
 
for ndim_counter in range(0,len(ndims)):
   for m in range(0,M+1):
      line1 = " " + str(all_mean[ndim_counter, m])
      target.write(line1)
      target.write("\n")
target.write("\n")

for ndim_counter in range(0,len(ndims)):
   for m in range(0,M+1):
      line1 = " " + str(all_var[ndim_counter, m])
      target.write(line1)
      target.write("\n")

target.write("\n")
target.close()










end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("*******************************************************************************************************")
