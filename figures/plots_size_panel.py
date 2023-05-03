# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator


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

line_label = ['Diag','vANN','PT','HF','Space']
linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,2]
linestyle_order = [2,0,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

# NUMBER OF PARTICLES
alphabet=["(a)","(b)","(c)","(d)","(e)"]
A=[2,3,4,5,6]
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
fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_V00.00e+00_SIG0_5.00e-01_Nopt_0000_Pretrain_False_DEVICE_cuda_BIT_float32_NBATCHES_010000.npz"

fig,ax = plt.subplots(nrows=num_plots,ncols=1,figsize=(7,3.5*num_plots),sharex=True)

for iy,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    print(Astr)
    Astr0=str(A_num_part).zfill(2)

    file="x2_" + Astr + "_particles.dat"

    # NONINTERACTING BASELINE
    efree=np.power(A_num_part,1)/2
    baseline=np.sqrt(efree)
    baseline=1.

# NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/data_s0.5_V/"
    fileML="PHYS_A" + Astr0 + "_NHID" + HH + "_NLAY" + LL + "_NDET"+ DD + fileML_app
    fname_ML= folder_ML + fileML
    if os.path.exists(fname_ML) :
        data = np.load(fname_ML) #load the PHYS.npz file
        v_ml = data['V0']
        e_ml = data['rms']
        #ax[iy].plot(v_ml,np.sqrt(e_ml),"-",color='#377eb8',label="vANN",zorder=3)
        iplot=1
        ax[iy].plot(v_ml,np.sqrt(e_ml),linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"
    # REAL SPACE SOLUTION FOR A=2
    if A_num_part == 2 :
        fname_2d= folder_2d_space + file
        if os.path.exists(fname_2d) :
            v_sp,e_sp = np.loadtxt(fname_2d,usecols=(0,2),unpack=True)
            iplot=4
            ax[iy].plot(v_sp,np.sqrt(e_sp)/baseline,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

# DIAGONALIZATION SOLUTION
    folder_diag="../Diagonalization/data_s0.5_V/"
    fname_diag= folder_diag + file
    if os.path.exists(fname_diag) :
        v_d,e_d = np.loadtxt(fname_diag,usecols=(0,1),unpack=True)
#        ax[iy].plot(v_d,np.sqrt(e_d)/baseline,"--",color='#FF0000',label="Diag",zorder=2)
        iplot=0
        ax[iy].plot(v_d,np.sqrt(e_d)/baseline,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])



    if ( A_num_part < 4 ) :
        folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
    else :
        folder_HF="../hartree_fock/data_s0.5_V/xL6.0_Nx240/"

    # HARTREE-FOCK SOLUTION
    fname_HF= folder_HF + file
    if os.path.exists(fname_HF) :
        v_hf,e_hf = np.loadtxt(fname_HF,usecols=(0,2),unpack=True)
        eeee=np.ones_like(v_hf)*efree

        ax[iy].plot(v_hf,np.sqrt(eeee)/baseline,":",color='#969696',zorder=0)

        iplot=3
        ax[iy].plot(v_hf,np.sqrt(e_hf)/baseline,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    ax[iy].set_ylabel(r"rms size, $\langle x^2 \rangle$")

    ax[iy].set_xlim([-20,20])
    ax[iy].set_ylim(ymin=0.5)

    ax[iy].tick_params(which='both',direction='in',top=True,right=True)
    ax[iy].tick_params(which='major',length=8)
    ax[iy].tick_params(which='minor',length=4)
    ax[iy].xaxis.set_ticks(np.arange(-20, 20.00001, 5))
    ax[iy].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[iy].yaxis.set_minor_locator(AutoMinorLocator(2))

    ax[iy].text(0.05, 0.75, alphabet[iy]+"\nA=" + Astr,transform=ax[iy].transAxes, fontsize=22)


ax[0].legend(frameon=False,handlelength=2,fontsize=20,handletextpad=1)
ax[num_plots-1].set_xlabel("Strength, $V_0$")
ax[0].set_title("$\sigma_0=0.5$")

file_figure="rms_size_panel.pdf"
print(file_figure)
plt.tight_layout()
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
