# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
#fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(0.75,0.5))

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
A=[2,3,4,5,6]

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

panel_letter = ["(a)","(b)"]

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"
for A_num_part in A :
    Astr=str(A_num_part)
    print(Astr)
    Astr0=str(A_num_part).zfill(2)

    file="OBDM_eigenvalues_" + Astr + "_particles.dat"

    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,8),sharex=True)


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

    for ipart in range(0,A_num_part):
        print(ipart)
        occ_ml=np.squeeze( xml[:,ipart] )
        occ_diag=np.squeeze( xdiag[ipart+1,:] )
        ax[0].plot(v0,occ_ml,".",color=c_full[ipart],label="NQS, $n_{"+str(ipart)+"}$",zorder=0)
        if(space) : ax[0].plot(v0sp,xspace[ipart+2,:].T,".",color=c_full[ipart],label="Space, $n_{"+str(ipart)+"}$",markeredgecolor="black",markerfacecolor="None")
        ax[0].plot(v0,occ_diag,"-",color=c_full[ipart],label="Diag $n_{"+str(ipart)+"}$")
        #if(space) : ax[0].plot(v0sp,xspace[ipart+2,:].T,"s",color=c_full[ipart],label="Space, $n_{"+str(ipart)+"}$",markersize=2)

    ax[0].set_ylim(0.85,1.02)
    ax[0].legend(frameon=False,fontsize=16)

    for ipart in range(A_num_part,14):
        occ_ml=np.squeeze( xml[:,ipart] )
        occ_diag=np.squeeze( xdiag[ipart+1,:] )
    #    ax[1].plot(v0,xdiag[ipart+1,:].T,"-",color=c_empt[ipart-A_num_part])
    #    ax[1].plot(v0sp,xspace[ipart+2,:].T,".",color=c_empt[ipart-A_num_part])
        ax[1].semilogy(v0,occ_ml,".",color=c_empt[ipart-A_num_part],label="ML, $n_{"+str(ipart)+"}$",zorder=0)
        ax[1].semilogy(v0,occ_diag,"-",color=c_empt[ipart-A_num_part]) #,label="Diag, $n_{"+str(ipart)+"}$")
        if(space) : ax[1].semilogy(v0sp,xspace[ipart+2,:].T,".",color=c_empt[ipart-A_num_part],label="Space, $n_{"+str(ipart)+"}$",markeredgecolor="black",markerfacecolor="None")

    ax[1].set(ylim =(1e-3, 0.2))
    ax[0].set_title("A=" + Astr + ", $\sigma_0=0.5$" )
    ax[1].set_xlabel("Strength, $V_0$")

    for ix in range(2) :
        ax[ix].set_ylabel("Occupations, $n_\\alpha$")
        ax[ix].set_xlim([-20,20])
        ax[ix].tick_params(which='both',direction='in',top=True,right=True)
        ax[ix].tick_params(which='major',length=8)
        ax[ix].tick_params(which='minor',length=4)
        #ax[ix].xaxis.set_minor_locator(AutoMinorLocator(5))
        #ax[ix].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[ix].text(0.05, 0.9, panel_letter[ix],transform=ax[ix].transAxes,fontsize=16)

    file_figure="occupations_" + Astr + ".pdf"
    print(file_figure)
    plt.savefig(file_figure, bbox_inches='tight')
    plt.close(fig)
