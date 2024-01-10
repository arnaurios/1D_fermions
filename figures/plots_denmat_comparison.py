# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.font_manager import FontProperties

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
         "ytick.major.right": "True"
         }

plt.rcParams.update(params)

size=16

V_strength=[-20,-10,0,10,20]
num_cols=len(V_strength)

# NUMBER OF PARTICLES
A=[2,3,4,5,6]
#A=[2,3]
num_plots=len(A)
print(num_plots)

methods=["ML","diag","HF"]
num_methods=len(methods)


print(num_methods)
zmax=2
zmin=-0.5


cmap = cm.RdBu_r

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

print("Network configuration: H%03i L%02i D%02i" % (H, L, D))
#    fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
fileML_mid="_Tanh_W4096_P010000_"

fileML_app="_S5.00e-01_Adam_PT_False_BATCH_010000_device_cuda_dtype_float32.npz"

folder_ML="../neural_network/data_s0.5_V/"



for AA,A_num_part in enumerate(A):

    fig, ax = plt.subplots(nrows=num_methods,ncols=len(V_strength),figsize=(len(V_strength)*2.25,num_methods*2.5),sharex=True)
#    fig, ax = plt.subplots(nrows=num_plots,ncols=len(V_strength),sharey=True,figsize=(num_cols*2,num_plots*2),sharex=True)

    for iy,method in enumerate(methods) : 
        Astr=str(A_num_part)
    
        Astr0=str(A_num_part).zfill(2)
    
        contours=[-0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
        norm = mcolors.TwoSlopeNorm(vcenter=0,vmin=zmin,vmax=zmax)
    
        if( method == "ML" ) :
            fileMLstart="PHYS_A" + Astr0 + "_H" + HH + "_L" + LL + "_D"+ DD + fileML_mid 
            
        elif( method == "HF" ) :
            file0="denmat_" + Astr + "_particles_V0="    
    
            if( A_num_part < 5 ) : 
                Nx=200; 
                xL=5.0;
                folder_HF="../hartree_fock/data/xL" + str(xL) +"_Nx" + str(Nx) + "/"
            else :
                Nx=240; 
                xL=6.0;
                folder_HF="../hartree_fock/data/xL" + str(xL) +"_Nx" + str(Nx) + "/"        
                   
        elif( method == "diag") : 
            file0="density_A_" + Astr + "_g_" 
            folder_diag="../Diagonalization/density_matrix/" 
            
        for iV0,V0 in enumerate(V_strength) :
    
    
    # NETWORK DATA 
            if( method == "ML" ) :
                V0str="{:.2e}".format(float(V0))
                print("{:.2e}".format(float(V0)))
    
                fileML_V0="V" + V0str #+ "_S" + Sstr
                fileML = fileMLstart + fileML_V0 + fileML_app
                fname_ML= folder_ML + fileML
                
                if os.path.exists(fname_ML) :
                    data = np.load(fname_ML) #load the PHYS.npz file
                    denmats = data['h_obdm'] #pass the key for what data you want
                    x = data['xedges_obdm']
                    y = data['yedges_obdm']
    
        #axis/dim =2 is the x and rho(x) data.
                    dd_ml= denmats #[:,:]
                    x_ml = x[:-1] + np.diff(x)/2.
                    y_ml = y[:-1] + np.diff(y)/2.
                    normalization=np.sum(np.diag(dd_ml)*np.diff(x))
                    dd_ml=dd_ml/normalization*A_num_part
                    normalization=np.sum(np.diag(dd_ml)*np.diff(x))
                    print("Norm",normalization)
    
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
                    dd_ml=data[:,2].squeeze().reshape((Nx,Nx))
    
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
                    dd_ml=data.squeeze().reshape((Nx,Nx))
    
                else :
                    sys.exit("Could not find file: " + fname_diag)
                    
    #        print(np.amin(dd_ml))
    #        print(np.amax(dd_ml))
    
            pc = ax[iy,iV0].pcolormesh(x_ml,y_ml,dd_ml, norm=norm, cmap=cmap,rasterized=True)        
            ax[iy,iV0].contour(x_ml,y_ml,dd_ml,contours, colors='k', linewidths=0.9)
            
            # AXES AND LIMITS
            ax[iy,iV0].axis("square")
            ax[iy,iV0].set_xlim([-5,5])
            ax[iy,iV0].xaxis.set_ticks(np.arange(-4, 5, 2))
            ax[iy,iV0].set_ylim([-5,5])
            ax[iy,iV0].yaxis.set_ticks(np.arange(-4, 5, 2))
    
    
            if( iy == 0) : ax[iy,iV0].set_title(r"$V_0=$" + str(V0) )
            if( iV0 == 0) :ax[iy,iV0].text(0.07, 0.82, "A=" + Astr+", "+ method,transform=ax[iy,iV0].transAxes, fontsize=18)
            if(iV0 > 0) : ax[iy,iV0].set_yticklabels([])

            # TICKS
            ax[iy,iV0].tick_params(which='both',direction='in',top=True,right=True)
            ax[iy,iV0].tick_params(which='major',length=8)
            ax[iy,iV0].tick_params(which='minor',length=4)
            ax[iy,iV0].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax[iy,iV0].yaxis.set_minor_locator(AutoMinorLocator(4))
    
            # COLORBAR 
            if( iV0 == num_cols-1 ) : 
                if( iy ==1 ) :
                    cax = ax[iy,num_cols-1].inset_axes([1.04, 0.0, 0.05, 1], transform=ax[iy,iV0].transAxes)
                    cax.tick_params(labelsize=14)
                    fig.colorbar(pc, ax=ax[iy,iV0], cax=cax, ticks=contours)
    
        ax[2,2].set_xlabel(r"Position, $x_1 = x'_1$",fontsize=size)
        ax[1,0].set_ylabel(r"Position, $x_2 = x'_2$",fontsize=size)
    
#    plt.tight_layout(pad=0.0)
    plt.tight_layout(h_pad=0.0,w_pad=0.0)
    file_figure="denmat_" + Astr + "_comp.pdf"
    print(file_figure)
    plt.savefig(file_figure, bbox_inches='tight')
    plt.close()
