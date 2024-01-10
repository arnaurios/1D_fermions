# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator)
from matplotlib.ticker import ScalarFormatter
import colorcet as cc
from scipy.special import comb

from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys


params = {'axes.linewidth': 1,
         'axes.labelsize': 22,
         'axes.titlesize': 22,
         'lines.markeredgecolor': "black",
         'lines.linewidth': 1.5,
         'xtick.labelsize': 16,
         'ytick.labelsize': 16,
         "xtick.direction": "in",
         "xtick.major.bottom": "True",
         "xtick.major.top": "True",
         "ytick.direction": "in",
         "ytick.major.right": "True",
         'font.size': 16
         }

plt.rcParams.update(params)

size=16
nminorticks=4

V_strength=[-20,-10,0,10,20]
#V_strength=[0,0,0,0,0]
num_cols=len(V_strength)

# NUMBER OF PARTICLES
A=[2,3,4,5,6]
num_plots=len(A)


fig, ax = plt.subplots(nrows=num_plots,ncols=len(V_strength),figsize=(num_cols*2,num_plots*2),sharex=True)
cmap = cm.Reds #cc.cm.fire#diverging_bwr_20_95_c54

cbars=[None]*num_cols

contourslow=np.arange(0.05,0.4,0.05)
contourshgh=np.arange(0.4,2.2,0.2)
contours=np.concatenate((contourslow,contourshgh))
#print(contours)

print("Method ML, HF or diag:")
method = sys.argv[1]


if( method == "ML" ) :
    # NETWORK PARAMETERS
    # NUMBER OF HIDDEN UNITS
    H=64
    HH=str(H).zfill(3)
    # NUMBER OF LAYERS
    L=2
    LL=str(L).zfill(2)
    # NUMBER OF DETERMINANTS
    D=1
    DD=str(D).zfill(2)
    
#    print("Network configuration: H%03i L%02i D%02i" % (H, L, D))
#    fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
#    folder_ML="../neural_network/data_s0.5_V/"
    print("Network configuration: H%03i L%02i D%02i" % (H, L, D))
#    fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
    fileML_mid="_Tanh_W4096_P010000_"
    
    fileML_app="_S5.00e-01_Adam_PT_False_BATCH_010000_device_cuda_dtype_float32.npz"
    
    folder_ML="../neural_network/data_s0.5_V/"


for iy,A_num_part in enumerate(A):
    
      norm = mcolors.PowerNorm(vmin=0,vmax=1,gamma=0.4)
      ccmax=0.
      cvals = [None]*num_plots
    
      Astr=str(A_num_part)
      Astr0=str(A_num_part).zfill(2)
    
      if( method == "ML" ) :
          fileMLstart="PHYS_A" + Astr0 + "_H" + HH + "_L" + LL + "_D"+ DD + fileML_mid 

#          fileML="PHYS_A" + Astr0 + "_NHID" + HH + "_NLAY" + LL + "_NDET"+ DD + fileML_app
 #         folder_ML="../neural_network/data_s0.5_V/"
  #        if( A_num_part ==2 ) : fileML="PHYS_A02_H064_L02_D01_Tanh_W4096_P001000_V0.00e+00_S5.00e-01_Adam_PT_False_BATCH_001000_device_cuda_dtype_float64.npz"
   #       if( A_num_part >=3 ) : fileML="PHYS_A03_H064_L02_D01_Tanh_W4096_P001000_V0.00e+00_S5.00e-01_Adam_PT_False_BATCH_001000_device_cuda_dtype_float64.npz"
   #       fname_ML= folder_ML + fileML

#          print(fileML)   
      elif( method == "HF" ) :
        file0="pairdist_" + Astr + "_particles_V0="    

        if( A_num_part < 5 ) : 
            Nx=200; 
            xL=5.0;
            folder_HF="../hartree_fock/data/xL" + str(xL) +"_Nx" + str(Nx) + "/"
        else :
            Nx=240; 
            xL=6.0;
            folder_HF="../hartree_fock/data/xL" + str(xL) +"_Nx" + str(Nx) + "/"        
          
      elif( method == "diag") : 
        file0="pair_correlation_A_" + Astr + "_g_" 
        folder_diag="../Diagonalization/pair_correlation/" 
            
    
      for iV0,V0 in enumerate(V_strength):
          # NETWORK DATA 
          if( method == "ML" ) :
              V0str="{:.2e}".format(float(V0))
              print("{:.2e}".format(float(V0)))
        
              fileML_V0="V" + V0str #+ "_S" + Sstr
              fileML = fileMLstart + fileML_V0 + fileML_app
              fname_ML= folder_ML + fileML
              
              if os.path.exists(fname_ML) :
                  data = np.load(fname_ML) #load the PHYS.npz file
        #                  denmats = data['h_tbdm'] #pass the key for what data you want
        #                  x = data['xedges_tbdm']
        #                  y = data['yedges_tbdm']
        
                  denmats = data['h_tbd'] #pass the key for what data you want
                  x = data['xedges_tbd']
                  y = data['yedges_tbd']
                
                  v0_idx=int(20+V0)
#                  dd_ml= denmats[v0_idx,:,:]
                  dd_ml1= denmats[:,:]
        #                  x_ml = x[v0_idx,:][:-1] + np.diff(x[v0_idx, :])/2.
        #                  y_ml = y[v0_idx,:][:-1] + np.diff(y[v0_idx, :])/2.
        
        
                  x_ml = x[:-1] + np.diff(x)/2.
                  y_ml = y[:-1] + np.diff(y)/2.
        
                  
                  print(x_ml.shape, y_ml.shape, dd_ml1.shape)
                  #normalization=np.sum(np.diag(dd_ml)*np.diff(x[v0_idx,:]))
                  
              else :
                  sys.exit("Could not find file: " + fname_ML)
        
            # HARTRE FOCK DATA
          elif( method == "HF" ) :
            file=file0 + str(V0) + ".0.dat"
        
            fname_HF= folder_HF + file
            print(fname_HF)
            if os.path.exists(fname_HF) :
                data = np.loadtxt(fname_HF) #load the PHYS.npz file
                x_ml=data[0:Nx,1]
                y_ml=data[0:Nx,1]
                dd_ml1=data[:,2].squeeze().reshape((Nx,Nx))
        #                x1,x2=np.meshgrid(x_ml,y_ml)
        #                dd_ml1=np.exp(-x1*x1-x2*x2)*np.power(x1-x2,2)/pi
        
            else :
                sys.exit("Could not find file: " + fname_HF)
        
        # DIAGONALIZATION DATA
          elif( method == "diag" ) :
            file=file0 + str(V0) + ".txt"
            fname_diag= folder_diag + file
            print(fname_diag)
            if os.path.exists(fname_diag) :
                data = np.loadtxt(fname_diag) #load the PHYS.npz file
                Nx=1001
                x_ml=np.linspace(-5,5,Nx,endpoint=True)
                y_ml=x_ml
                dd_ml1=data.squeeze().reshape((Nx,Nx))
        
          integral = np.trapz(np.trapz(dd_ml1, y_ml, axis=0), x_ml, axis=0)
          print(integral)
          print("nCr: ",comb(A_num_part,2))
          dd_ml=dd_ml1/integral*comb(A_num_part, 2) #integral-norm of density matrix

    #       print(np.amax(np.amax(dd_ml)))
   
          pc = ax[iy,iV0].pcolormesh(x_ml,y_ml,dd_ml,cmap=cmap, norm=norm,rasterized=True)
          ax[iy,iV0].contour(x_ml,y_ml,dd_ml, contours, linestyles='--', colors='#999999', linewidths=0.7)
          ax[iy,iV0].contour(x_ml,y_ml,dd_ml, contourslow, colors='k', linewidths=0.7)
    
          # AXES AND LIMITS
          ax[iy,iV0].axis("square")
          ax[iy,iV0].set_xlim([-5,5])
          ax[iy,iV0].xaxis.set_ticks(np.arange(-4, 5, 2))
          ax[iy,iV0].set_ylim([-5,5])
          ax[iy,iV0].yaxis.set_ticks(np.arange(-4, 5, 2))
    
          if(iV0 > 0) : ax[iy,iV0].set_yticklabels([])
    
          if( iy == 0) : ax[iy,iV0].set_title(r"$V_0=$" + str(V0) )
          if( iV0 == 0) :ax[iy,iV0].text(0.07, 0.82, "A=" + Astr,transform=ax[iy,iV0].transAxes, fontsize=18)
    
          #ticks
          ax[iy,iV0].tick_params(which='both',direction='in',top=True,right=True)
          ax[iy,iV0].tick_params(which='major',length=8)
          ax[iy,iV0].tick_params(which='minor',length=4)
          ax[iy,iV0].xaxis.set_minor_locator(AutoMinorLocator(4))
          ax[iy,iV0].yaxis.set_minor_locator(AutoMinorLocator(4))
    
         
          
          #print("lim: ",pc.get_clim())
          cmin, cmax = pc.get_clim()
          ccmax = max(np.abs(cmin), np.abs(cmax))
          if(iV0 == num_cols-1):
            cbars[iV0] = pc
            cax = ax[iy,num_cols-1].inset_axes([1.04, 0.0, 0.05, 1], transform=ax[iy,iV0].transAxes) #num_cols-1
            fig.colorbar(cbars[iV0], ax=ax[iy,num_cols-1], cax=cax)#, ticks=contours)
            
      ax[4,2].set_xlabel(r"Position, $x_1 = x'_1$",fontsize=size)
      ax[2,0].set_ylabel(r"Position, $x_2 = x'_2$",fontsize=size)
      
  #ax[iy,2].legend(frameon=False,fontsize=12,handlelength=1.5)#size was 12

#            if( iV0 == 0) :ax[ix,iV0].text(0.07, 0.82, "A=" + Astr,transform=ax[ix,iV0].transAxes, fontsize=18)  
#  ax[iy,0].text(0.4,0.9, "A=" + Astr,# +"\n$\sigma_0=0.5$",
 #     horizontalalignment='right',
  #    verticalalignment='top',
   #   transform=ax[iy,0].transAxes,size=18)#was18

plt.tight_layout(h_pad=0.0,w_pad=0.0)

file_figure="2body_denmat_panel_" + method + ".pdf"
print(file_figure)
plt.savefig(file_figure, bbox_inches='tight')
#plt.show()