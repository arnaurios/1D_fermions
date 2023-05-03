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

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 20,
         'axes.titlesize': 20,
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

#plt.rcParams.update({'font.size': 18})

plt.rcParams.update(params)

V_strength=[-20,-10,0,10,20]
num_cols=len(V_strength)

# NUMBER OF PARTICLES
A=[2,3,4,5,6]
ymax=[2,2,2.25,2.5,2.7]

ymax=[2,2,2,2,2]
ymin=[-0.5,-0.5,-0.6,-0.6,-0.6]


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
print(HH,LL,DD)

cmap = cm.coolwarm

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
for ix,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    print(Astr)

    Astr0=str(A_num_part).zfill(2)
    print(Astr0)

    #contours=[-0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    contours=[-0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
    print(contours)
    norm = mcolors.TwoSlopeNorm(vcenter=0,vmin=ymin[ix],vmax=ymax[ix])
    fig, ax = plt.subplots(nrows=1,ncols=len(V_strength),sharey=True,figsize=(num_cols*3,3))

    file0="density_" + Astr + "_particles_V0="

    fileML="PHYS_A" + Astr0 + "_NHID" + HH + "_NLAY" + LL + "_NDET"+ DD + fileML_app
    print(fileML)

    folder_diag="../Diagonalization/data_s0.5_V/"
    if ( A_num_part < 4 ) :
        folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
    else :
        folder_HF="../hartree_fock/data_s0.5_V/xL6.0_Nx240/"

    folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"

    for iV0,V0 in enumerate(V_strength) :
        file=file0 + str(V0) + ".0.dat"

# NETWORK DATA SHOULD GO HERE
        folder_ML="../neural_network/data_s0.5_V/"
        fname_ML= folder_ML + fileML
        if os.path.exists(fname_ML) :
            data = np.load(fname_ML) #load the PHYS.npz file
            denmats = data['h_obdm'] #pass the key for what data you want
            x = data['xedges_obdm']
            y = data['yedges_obdm']

    #axis/dim =2 is the x and rho(x) data.
            v0_idx=int(20+V0)
            dd_ml= denmats[v0_idx,:,:]
            x_ml = x[v0_idx,:][:-1] + np.diff(x[v0_idx, :])/2.
            y_ml = y[v0_idx,:][:-1] + np.diff(y[v0_idx, :])/2.

            #print(np.diff(x[v0_idx,:]))
            normalization=np.sum(np.diag(dd_ml)*np.diff(x[v0_idx,:]))
            dd_ml=dd_ml/normalization*A_num_part
            normalization=np.sum(np.diag(dd_ml)*np.diff(x[v0_idx,:]))
            print("Norm",normalization)
            print(np.amin(dd_ml))
            print(np.amax(dd_ml))

            pc=ax[iV0].contourf(x_ml,y_ml,dd_ml,contours,norm=norm,cmap=cmap)#,levels=np.arange(ymin[ix],ymax[ix],0.1), norm=norm, cmap=cmap)
            #pc = ax[iV0].pcolormesh(x_ml,y_ml,dd_ml, norm=norm, cmap=cmap)
            ax[iV0].contour(x_ml,y_ml,dd_ml,contours, colors='k')

            ax[iV0].axis("square")
            ax[iV0].set_xlim([-5,5])
            ax[iV0].xaxis.set_ticks(np.arange(-4, 5, 2))
            ax[iV0].set_ylim([-5,5])
            ax[iV0].yaxis.set_ticks(np.arange(-4, 5, 2))

        # REAL SPACE SOLUTION FOR A=2
#        if A_num_part == 2 :
#            fname_2d= folder_2d_space + file
#            if os.path.exists(fname_2d) :
#                print(fname_2d)
#                x_sp,d_sp = np.loadtxt(fname_2d,usecols=(0,1),unpack=True)
#                ax[iV0].plot(x_sp,d_sp,":",color='#ff7f00',label="Space",zorder=4)

            ax[iV0].set_xlabel("Position, $x_1$")
            ax[iV0].set_title("$A=$"+Astr+", $V_0=$" + str(V0) )
            ax[iV0].tick_params(which='both',direction='in',top=True,right=True)
            ax[iV0].tick_params(which='major',length=8)
            ax[iV0].tick_params(which='minor',length=4)
            ax[iV0].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax[iV0].yaxis.set_minor_locator(AutoMinorLocator(4))

        ax[0].set_ylabel("Position, $x_2$")
        cax = ax[num_cols-1].inset_axes([1.04, 0.0, 0.05, 1], transform=ax[num_cols-1].transAxes)
        fig.colorbar(pc, ax=ax[num_cols-1], cax=cax, ticks=contours)


    plt.tight_layout(pad=-0.5)
    file_figure="denmat_" + Astr + ".pdf"
    print(file_figure)
    plt.savefig(file_figure, bbox_inches='tight')
    plt.close()
