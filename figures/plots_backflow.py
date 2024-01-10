# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


params = {'axes.linewidth': 1.4,
         'axes.labelsize': 19,
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

A=[2,6]

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

print("Pos or neg value + -:")
pos_neg = sys.argv[1]
pn=pos_neg 
sign=-1.
if(pos_neg == "+") : pn=""; sign=1.;

V0=sign*20
V0str="{0:+.2g}".format(float(V0))

# NETWORK DATA SHOULD GO HERE
folder_ML="../neural_network/backflow/"
fileML_app="_Tanh_W4096_P010000_V" + pn + "2.00e+01_S5.00e-01_Adam_PT_False_device_cuda_dtype_float32.csv"

fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,8),sharex=True)

line_label = ['Diag','NQS','PT','HF','Space']
linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,2]
linestyle_order = [10,3,0,10,4]
linestyle_dashes = ['--','-','','-.','.']


for ix,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    print(Astr)
    Astr0=str(A_num_part).zfill(2)


    file_ML="A0" + Astr + "_H" + HH + "_L" + LL + "_D" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    
    df = pd.read_csv(fname_ML)
    iplot=1
    ax[ix].plot(df['epoch'], df['energy_mean'],linestyle_dashes[iplot],color=linestyle_color[iplot],label='NQS (with backflow)',zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
    
#    ax[ix].plot(df['epoch'], df['energy_mean'], label='NQS (w/ backflow)')

    fname_ML= folder_ML + "no_backflow_" + file_ML
    df = pd.read_csv(fname_ML)   
    ax[ix].plot(df['epoch'], df['energy_mean'], label='NQS (without backflow)',color="C1")

    epoc=[0,1e5]

    # DIAGONALIZATION SOLUTION
    file="Hamiltonian_eigenvalues" + Astr + "_particles.dat"
    folder_diag="../Diagonalization/data_s0.5_V/"
    # DIAGONALIZATION SOLUTION
    fname_diag= folder_diag + file
    if os.path.exists(fname_diag) :
        v_d,e_d = np.loadtxt(fname_diag,usecols=(0,1),unpack=True)
        indx=list(v_d).index(sign*20.)
        ediag=e_d[indx]
        
    if( sign==1. or (sign==-1. and A_num_part<5) ) : 
        iplot=0
        ax[ix].plot(epoc,np.ones_like(epoc)*ediag,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])


    # HARTREE-FOCK SOLUTION
    if ( A_num_part < 4 ) :
        folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
    else :
        folder_HF="../hartree_fock/data_s0.5_V/xL6.0_Nx240/"
    fname_HF= folder_HF + file
    if os.path.exists(fname_HF) :
        v_hf,e_hf = np.loadtxt(fname_HF,usecols=(0,2),unpack=True)
        indx=list(v_hf).index(sign*20.)
        ehf=e_hf[indx]

    iplot=3
    ax[ix].plot(epoc,np.ones_like(epoc)*ehf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

                                                                                                                                                               
    ax[ix].set_ylabel("Ground state energy, $E$")
    ax[ix].set_title("A=" + Astr + ", $V_0=$" + pn + "$20$, $\sigma_0=0.5$" )
    ax[1].set_xlabel("Epochs")
    ax[ix].set_xlim([0,100000])



ax[0].legend(frameon=False,fontsize=16,handlelength=3.5)

file_figure="backflow_V" + V0str + ".pdf"
print(file_figure)
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
