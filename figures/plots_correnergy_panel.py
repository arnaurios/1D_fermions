# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.font_manager import FontProperties

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 24,
         'axes.titlesize': 24,
         'lines.markeredgecolor': "black",
         'lines.linewidth': 1.5,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20,
         "xtick.direction": "in",
         "xtick.major.bottom": "True",
         "xtick.major.top": "True",
         "ytick.direction": "in",
         "ytick.major.right": "True"
         }

plt.rcParams.update(params)

line_label = ['Diag','NQS','PT','HF','Space']
linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,2]
linestyle_order = [2,0,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

# NUMBER OF PARTICLES
#alphabet=["(a)","(b)","(c)","(d)","(e)"]
#A=[2,3,4,5,6]

alphabet=["(a)","(b)","(c)","(d)"]
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

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_SIG0_5.00e-01_NBATCHES_010000.txt"

fig,ax = plt.subplots(nrows=num_plots,ncols=1,figsize=(7,3.5*num_plots),sharex=True)

for iy,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    AA=Astr.zfill(2)
    print(Astr,AA)
    file="Hamiltonian_eigenvalues" + Astr + "_particles.dat"

    # NONINTERACTING BASELINE
    efree=np.power(A_num_part,2)/2

    if ( A_num_part < 4 ) :
        folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
    else :
        folder_HF="../hartree_fock/data_s0.5_V/xL6.0_Nx240/"

    # HARTREE-FOCK SOLUTION
    fname_HF= folder_HF + file
    if os.path.exists(fname_HF) :
        v_hf,e_hf = np.loadtxt(fname_HF,usecols=(0,2),unpack=True)

        eeee=np.zeros_like(v_hf)
        ax[iy].plot(v_hf,eeee,":",color='#969696',zorder=0)

    # NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/Energy/"
    file_ML="energy_A" + AA + "_NHID" + HH + "_NLAY" + LL + "_NDET" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    print(fname_ML)
    if os.path.exists(fname_ML) :
        mlx = np.loadtxt(fname_ML,usecols=(0,1,2),delimiter=" ")
        hbarom=40*np.power(A_num_part,-1./3.)
        v_ml=mlx[:,0]
        e_ml=mlx[:,1]
        ee_ml=mlx[:,2]
        iplot=1
        ax[iy].fill_between(v_ml,e_ml-e_hf-ee_ml,e_ml-e_hf+ee_ml,color=linestyle_color[iplot],alpha=0.8,zorder=3)
        ax[iy].plot(v_ml,e_ml-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    # REAL SPACE SOLUTION FOR A=2
    folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"
    if A_num_part == 2 :
        fname_2d= folder_2d_space + file
        if os.path.exists(fname_2d) :
            v_sp,e_sp = np.loadtxt(fname_2d,usecols=(0,2),unpack=True)
            iplot=4
            ax[iy].plot(v_sp,e_sp-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    folder_diag="../Diagonalization/data_s0.5_V/"
    # DIAGONALIZATION SOLUTION
    fname_diag= folder_diag + file
    if os.path.exists(fname_diag) :
        v_d,e_d = np.loadtxt(fname_diag,usecols=(0,1),unpack=True)
        iplot=0
        ax[iy].plot(v_d,e_d-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    ax[iy].set_xlim([-20,20])
    ax[iy].set_ylim([-1,0])

    ax[iy].tick_params(which='both',direction='in',top=True,right=True)
    ax[iy].tick_params(which='major',length=8)
    ax[iy].tick_params(which='minor',length=4)
    ax[iy].xaxis.set_ticks(np.arange(-20, 20.00001, 5))
    ax[iy].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[iy].yaxis.set_minor_locator(AutoMinorLocator(2))

    ax[iy].text(0.85, 0.75, alphabet[iy]+"\nA=" + Astr,transform=ax[iy].transAxes, fontsize=22)

    ax[0].legend(frameon=False,handlelength=2,fontsize=20,handletextpad=1)

ax[2].set_ylabel("Correlation energy, $E-E_{HF}$")

ax[num_plots-1].set_xlabel("Strength, $V_0$")
ax[0].set_title("$\sigma_0=0.5$")

file_figure="correlation_energy_panel.pdf"
print(file_figure)
plt.tight_layout()
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
