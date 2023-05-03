# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'lines.markersize': 10,
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

# NUMBER OF PARTICLES
A=[3,4,5,6]
num_plots=len(A)
print(num_plots)


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

panel_letter = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)"]

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"

fig,ax = plt.subplots(nrows=num_plots,ncols=2,figsize=(9,3*num_plots),sharex=True)

for iA,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    print(Astr)
    Astr0=str(A_num_part).zfill(2)

    file="OBDM_eigenvalues_" + Astr + "_particles.dat"

    fileML="PHYS_A" + Astr0 + "_NHID" + HH + "_NLAY" + LL + "_NDET"+ DD + fileML_app
    print(fileML)
# NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/data_s0.5_V/"
    fname_ML= folder_ML + fileML
    if os.path.exists(fname_ML) :
        data = np.load(fname_ML) #load the PHYS.npz file
        v_ml = data['V0']
        eigenvalues = data['eigenvalues']
        xml = eigenvalues

    # DIAGONALIZATION SOLUTION
    folder_diag="../Diagonalization/data_s0.5_V/"
    fname_diag= folder_diag + file
    if os.path.exists(fname_diag) :
        xdiag = np.loadtxt(fname_diag,unpack=True)
        v0=np.squeeze( xdiag[0,:].T )

    # SPACE SOLUTION
    folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"
    fname_diag = folder_2d_space + file
    if os.path.exists(fname_diag ) :
        space=True
        xspace = np.loadtxt(fname_diag,unpack=True)
        v0sp=xspace[0,:].T
    else :
        space=False

    c_full = plt.cm.Blues(np.linspace(0.5,1,A_num_part))
    c_empt = plt.cm.Reds(np.linspace(1,0.2,15-A_num_part))

    ipart=0
    occ_ml=np.squeeze( xml[:,ipart] )
    occ_diag=np.squeeze( xdiag[ipart+1,:] )
    ax[iA,0].plot(v0,occ_ml,".",color=c_full[ipart],label=r"NQS, $n_{\alpha}$",zorder=0)
    if(space) : ax[iA,0].plot(v0sp,xspace[ipart+2,:].T,".",color=c_full[ipart],label=r"Space, $n_{\alpha}$",markeredgecolor="black",markerfacecolor="None")
    ax[iA,0].plot(v0,occ_diag,"-",color=c_full[ipart],label=r"Diag, $n_{\alpha}$")

    for ipart in range(1,A_num_part):
        print(ipart)
        occ_ml=np.squeeze( xml[:,ipart] )
        occ_diag=np.squeeze( xdiag[ipart+1,:] )
        ax[iA,0].plot(v0,occ_ml,".",color=c_full[ipart],zorder=0)
        if(space) : ax[iA,0].plot(v0sp,xspace[ipart+2,:].T,".",color=c_full[ipart],markeredgecolor="black",markerfacecolor="None")
        ax[iA,0].plot(v0,occ_diag,"-",color=c_full[ipart])
        #if(space) : ax[0].plot(v0sp,xspace[ipart+2,:].T,"s",color=c_full[ipart],label="Space, $n_{"+str(ipart)+"}$",markersize=2)

    ax[iA,0].set_ylim(0.85,1.06)
    ax[iA,0].legend(frameon=False,fontsize=16)
    ax[iA,0].text(0.03, 0.87, panel_letter[iA]+"  A=" + Astr,transform=ax[iA,0].transAxes, fontsize=22)
    ax[iA,0].set_ylabel(r"Occupations, $n_\alpha$")
    ax[iA,0].yaxis.set_minor_locator(AutoMinorLocator(5))

    for ipart in range(A_num_part,14):
        occ_ml=np.squeeze( xml[:,ipart] )
        occ_diag=np.squeeze( xdiag[ipart+1,:] )
    #    ax[1].plot(v0,xdiag[ipart+1,:].T,"-",color=c_empt[ipart-A_num_part])
    #    ax[1].plot(v0sp,xspace[ipart+2,:].T,".",color=c_empt[ipart-A_num_part])
        ax[iA,1].semilogy(v0,occ_ml,".",color=c_empt[ipart-A_num_part],label="ML, $n_{"+str(ipart)+"}$",zorder=0)
        ax[iA,1].semilogy(v0,occ_diag,"-",color=c_empt[ipart-A_num_part]) #,label="Diag, $n_{"+str(ipart)+"}$")
        if(space) : ax[iA,1].semilogy(v0sp,xspace[ipart+2,:].T,".",color=c_empt[ipart-A_num_part],label="Space, $n_{"+str(ipart)+"}$",markeredgecolor="black",markerfacecolor="None")

    ax[iA,1].set(ylim =(5e-3, 0.3))
    ax[iA,1].text(0.03, 0.87, panel_letter[iA+num_plots]+"  A=" + Astr,transform=ax[iA,1].transAxes, fontsize=22)

    
    for ix in range(2) :   
        ax[iA,ix].set_xlim([-20,20])
        ax[iA,ix].tick_params(which='both',direction='in',top=True,right=True)
        ax[iA,ix].tick_params(which='major',length=8)
        ax[iA,ix].tick_params(which='minor',length=4)
        ax[iA,ix].xaxis.set_minor_locator(AutoMinorLocator(5))
        
       
        #ax[iA,ix].text(0.05, 0.9, panel_letter[ix],transform=ax[iA,ix].transAxes,fontsize=16)

ax[num_plots-1,0].set_xlabel("Strength, $V_0$")
ax[num_plots-1,1].set_xlabel("Strength, $V_0$")

file_figure="occupations_panel.pdf"
print(file_figure)
plt.tight_layout(pad=1)
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
