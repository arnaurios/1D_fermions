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
         'xtick.labelsize': 20,
         'ytick.labelsize': 20,
         "xtick.direction": "in",
         "xtick.major.bottom": "True",
         "xtick.major.top": "True",
         "ytick.direction": "in",
         "ytick.major.right": "True"
         }

plt.rcParams.update(params)

linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,3]
line_label = ['Diag','NQS','PT','HF','Space','PT1','PT2','PT3']
linestyle_order = [2,3,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

panel_letter = ["(a)","(b)","(c)"]

#fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,8),sharex=True)
fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(7,12),sharex=True)

# NETWORK PARAMETERS
H=64
HH=str(H).zfill(3)
L=2
LL=str(L).zfill(2)
D=1
DD=str(D).zfill(2)
print(HH,LL,DD)

fileML_app="_FUNC_Tanh_NW004096_OPTIM_Adam_SIG0_5.00e-01_NBATCHES_010000.txt"

A_num_part=2
Astr=str(A_num_part)
AA=Astr.zfill(2)
print(Astr)
# TARGET STATE nstate=0; nstate=1 1st excited; etc
nstate=0

file_space="Hamiltonian_eigenvalues" + Astr + "_particles.dat"
file0_PT="Hamiltonian_eigenvalues" + Astr + "_particles"

folder_2d_space="../A2_space_solution/data_s0.5_V/xL5.0_Nx200/"
folder_PT="../A2_perturbation_theory/data_s0.5_V/"

file="Hamiltonian_eigenvalues" + Astr + "_particles.dat"

# NONINTERACTING BASELINE
efree=np.power(A_num_part,2)/2

# HARTREE-FOCK SOLUTION
folder_HF="../hartree_fock/data_s0.5_V/xL5.0_Nx200/"
fname_HF= folder_HF + file
if os.path.exists(fname_HF) :
    v_hf,e_hf = np.loadtxt(fname_HF,usecols=(0,2),unpack=True)
    eeee=np.ones_like(v_hf)*efree
    ax[1].plot(v_hf,eeee,":",color='#969696',zorder=0)
    
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
    for ix in range(2) :
        ax[ix].fill_between(v_ml,e_ml-ee_ml,e_ml+ee_ml,color=linestyle_color[iplot],alpha=0.8,zorder=3)
        ax[ix].plot(v_ml,e_ml,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    ax[2].fill_between(v_ml,e_ml-e_hf-ee_ml,e_ml-e_hf+ee_ml,color=linestyle_color[iplot],alpha=0.8,zorder=3)
    ax[2].plot(v_ml,e_ml-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

if os.path.exists(fname_HF) :
    iplot=3
    ax[1].plot(v_hf,e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])


# REAL SPACE SOLUTION FOR A=2
fname_2d= folder_2d_space + file_space
if os.path.exists(fname_2d) :
    v_sp,e_sp = np.loadtxt(fname_2d,usecols=(0,2+nstate),unpack=True)
    iplot=4
    ax[1].plot(v_sp,e_sp,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot],markersize=10)
    ax[2].plot(v_sp,e_sp-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot],markersize=10)


# DIAGONALIZATION SOLUTION
folder_diag="../Diagonalization/data_s0.5_V/"
fname_diag= folder_diag + file
if os.path.exists(fname_diag) :
    v_d,e_d = np.loadtxt(fname_diag,usecols=(0,1),unpack=True)
    iplot=0
    ax[1].plot(v_d,e_d,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
    ax[2].plot(v_d,e_d-e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

# PT SOLUTION ORDER 1
fname_PT1= folder_PT + file0_PT + "_order1.dat"
if os.path.exists(fname_PT1) :
    v_PT1,e_PT1 = np.loadtxt(fname_PT1,usecols=(0,2+nstate),unpack=True)
    eeee=np.ones_like(v_PT1)*efree
    ax[0].plot(v_PT1,eeee,":",color='#969696')
    ax[0].plot(v_PT1,e_PT1,"-", dashes=[6, 2, 1, 2, 1, 2],color='#74c476',label="PT1")

# PT SOLUTION ORDER 2
fname_PT2= folder_PT + file0_PT + "_order2.dat"
if os.path.exists(fname_PT2) :
    v_PT2,e_PT2 = np.loadtxt(fname_PT2,usecols=(0,2+nstate),unpack=True)
    ax[0].plot(v_PT2,e_PT2,"-.",color='#31a354',label="PT2")

# PT SOLUTION ORDER 3
fname_PT3= folder_PT + file0_PT + "_order3.dat"
if os.path.exists(fname_PT3) :
    v_PT3,e_PT3 = np.loadtxt(fname_PT3,usecols=(0,2+nstate),unpack=True)
    ax[0].plot(v_PT3,e_PT3,"--",color='#006d2c',label="PT3")



ax[0].legend(frameon=False,handlelength=3.2,fontsize=20,handletextpad=0.25,ncols=2)
ax[1].legend(frameon=False,handlelength=3.2,fontsize=20,handletextpad=0.25,ncols=2)
ax[2].set_xlabel("Strength, $V_0$")
ax[0].set_title("A=" + Astr + ", $\sigma_0=0.5$" )

for ix in range(2) :
    ax[ix].set_ylabel("Ground state energy, $E$")
    ax[ix].set_ylim([-3,3.5])
    ax[ix].set_xlim([-20,20])
    ax[ix].yaxis.set_minor_locator(AutoMinorLocator(4))

for ix in range(3) :
    ax[ix].tick_params(which='both',direction='in',top=True,right=True)
    ax[ix].tick_params(which='major',length=8)
    ax[ix].tick_params(which='minor',length=4)
    ax[ix].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[ix].text(0.05, 0.9, panel_letter[ix],transform=ax[ix].transAxes,fontsize=20)

ax[2].set_ylim([-0.6,0])
ax[2].set_ylabel("Correlation energy, $E-E_{HF}$")


file_figure="energy_A" + Astr + "_panel.pdf"
print(file_figure)
#plt.show()

plt.tight_layout()
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
