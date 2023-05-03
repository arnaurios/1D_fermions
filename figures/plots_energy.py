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

line_label = ['Diag','vANN','PT','HF','Space']
linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,2]
linestyle_order = [2,3,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

# NUMBER OF PARTICLES
A=[2,3,4,5,6]

# NETWORK PARAMETERS
H=64
HH=str(H).zfill(3)
L=2
LL=str(L).zfill(2)
D=1
DD=str(D).zfill(2)
print(HH,LL,DD)

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_SIG0_5.00e-01_NBATCHES_010000.txt"

for A_num_part in A :
    Astr=str(A_num_part)
    AA=Astr.zfill(2)
    print(Astr,AA)
    file="Hamiltonian_eigenvalues" + Astr + "_particles.dat"

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,4))

    # NONINTERACTING BASELINE
    efree=np.power(A_num_part,2)/2

    # NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/Energy/"
    file_ML="energy_A" + AA + "_NHID" + HH + "_NLAY" + LL + "_NDET" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    print(fname_ML)
    if os.path.exists(fname_ML) :
        mlx = np.loadtxt(fname_ML,usecols=(0,1,2),delimiter=" ")
        v_ml=mlx[:,0]
        e_ml=mlx[:,1]
        ee_ml=mlx[:,2]
        iplot=1
        ax.fill_between(v_ml,e_ml-ee_ml,e_ml+ee_ml,color=linestyle_color[iplot],alpha=0.8,zorder=3)
        ax.plot(v_ml,e_ml,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    # REAL SPACE SOLUTION FOR A=2
    folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"
    if A_num_part == 2 :
        fname_2d= folder_2d_space + file
        if os.path.exists(fname_2d) :
            v_sp,e_sp = np.loadtxt(fname_2d,usecols=(0,2),unpack=True)
            iplot=4
            ax.plot(v_sp,e_sp,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    folder_diag="../Diagonalization/data_s0.5_V/"
    # DIAGONALIZATION SOLUTION
    fname_diag= folder_diag + file
    if os.path.exists(fname_diag) :
        v_d,e_d = np.loadtxt(fname_diag,usecols=(0,1),unpack=True)
        iplot=0
        ax.plot(v_d,e_d,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    if ( A_num_part < 4 ) :
        folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
    else :
        folder_HF="../hartree_fock/data_s0.5_V/xL6.0_Nx240/"

    # HARTREE-FOCK SOLUTION
    fname_HF= folder_HF + file
    if os.path.exists(fname_HF) :
        v_hf,e_hf = np.loadtxt(fname_HF,usecols=(0,2),unpack=True)

        eeee=np.ones_like(v_hf)*efree
        ax.plot(v_hf,eeee,":",color='#969696',zorder=0)
        iplot=3
        ax.plot(v_hf,e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    ax.legend(frameon=False,handlelength=2.7,ncol=2,fontsize=16,handletextpad=1)

    ax.set_xlabel("Strength, $V_0$")
    ax.set_ylabel("Ground state energy, $E$")
    ax.set_title("A=" + Astr + ", $\sigma_0=0.5$" )
    ax.set_xlim([-20,20])

    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.tick_params(which='major',length=8)
    ax.tick_params(which='minor',length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    file_figure="energy_" + Astr + ".pdf"
    print(file_figure)
    plt.savefig(file_figure, bbox_inches='tight')
    plt.close(fig)
