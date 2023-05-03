# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.font_manager import FontProperties

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

plt.rcParams.update(params)
V_strength=[-20,-10,0,10,20]
num_cols=len(V_strength)

# NUMBER OF PARTICLES
#A=[2,3,4,5,6]
#ymax=[1.75,2,2.25,2.5,3]

# NUMBER OF PARTICLES
A=[3,4,5,6]
ymax=[2,2.25,2.5,3]

num_plots=len(A)
print(num_plots)

linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [1.5,2.5,1.5,1.5,1.5]
line_label = ['Diag','NQS','PT','HF','Space']
linestyle_order = [2,5,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

fig, ax = plt.subplots(nrows=num_plots,ncols=len(V_strength),figsize=(2.2*num_cols,2.5*num_plots))

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


fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
for iy,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    print(Astr)

    Astr0=str(A_num_part).zfill(2)
    print(Astr0)

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
            density_1b = data['one_densitys'] #pass the key for what data you want
    #axis/dim =2 is the x and rho(x) data.
            v0_idx=int(20+V0)
            x_ml = density_1b[v0_idx,:,0]
    #v0_idx is the particular V0 value where v0_idx=0 is -20 and v0_idx=40 is +20 (in general v0_idx = int(V0 + 20) for a given V0 value)
            d_ml = density_1b[v0_idx,:,1]
            iplot=1
            ax[iy,iV0].plot(x_ml,d_ml,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

        # REAL SPACE SOLUTION FOR A=2
        if A_num_part == 2 :
            fname_2d= folder_2d_space + file
            if os.path.exists(fname_2d) :
                print(fname_2d)
                x_sp,d_sp = np.loadtxt(fname_2d,usecols=(0,1),unpack=True)
                iplot=4
                ax[iy,iV0].plot(x_sp,d_sp,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot],markersize=9,markeredgewidth=1,markeredgecolor="black")

    # DIAGONALIZATION SOLUTION
        fname_diag= folder_diag + file
        if os.path.exists(fname_diag) :
            x_d,d_d = np.loadtxt(fname_diag,usecols=(0,2),unpack=True)
            iplot=0
            ax[iy,iV0].plot(x_d,d_d,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    # HARTREE-FOCK SOLUTION
        fname_HF= folder_HF + file
        if os.path.exists(fname_HF) :
            x_hf,d_hf = np.loadtxt(fname_HF,usecols=(0,1),unpack=True)
            iplot=3
            ax[iy,iV0].plot(x_hf,d_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

        ax[iy,iV0].set_xlim([-6,6])
        if(iV0 == num_cols-1) : ax[iy,iV0].set_xlim([-6,6.0001])
        ax[iy,iV0].set_ylim([0,ymax[iy]])

        start, end = ax[iy,iV0].get_xlim()
        ax[iy,iV0].xaxis.set_ticks(np.arange(start, end, 2))

        ax[iy,iV0].tick_params(which='both',direction='in',top=True,right=True)
        ax[iy,iV0].tick_params(which='major',length=8)
        ax[iy,iV0].tick_params(which='minor',length=4)
#        ax[iy,iV0].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[iy,iV0].yaxis.set_minor_locator(AutoMinorLocator(4))

        if(iy < num_plots-1) : ax[iy,iV0].set_xticklabels([])
        if(iV0 > 0) : ax[iy,iV0].set_yticklabels([])

        if(iy==0) : ax[iy,iV0].set_title("$V_0=$" + str(V0) )
        if(iy==num_plots-1) : ax[iy,iV0].set_xlabel("Position, $x$")

    ax[iy,0].set_ylabel("Density, $n(x)$")

    ax[iy,4].text(0.95,0.95, "A=" + Astr +"\n$\sigma_0=0.5$",
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[iy,4].transAxes,size=18)

ax[0,2].legend(frameon=False,fontsize="16",handlelength=2.5)

plt.tight_layout(h_pad=-.1,w_pad=-0.5)
file_figure="density_panel.pdf"
print(file_figure)
plt.savefig(file_figure, bbox_inches='tight')
plt.close()
