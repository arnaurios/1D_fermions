 # coding: utf-8
import sys
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

line_label = ['Diag','NQS','PT','HF','Space']
linestyle_color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
linestyle_width = [2,4,2,2,2]
linestyle_order = [-2,3,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,4))


#pos_neg=input("Pos or neg value + -:")
print("Pos or neg value + -:")
pos_neg = sys.argv[1]
pn=pos_neg 
if(pos_neg == "+") : pn=""

A_num_part=2
Astr=str(A_num_part)
print(Astr)

# NUMBER OF STATES
nstate=1

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

file_space="Hamiltonian_eigenvalues" + Astr + "_particles.dat"
file0_PT="Hamiltonian_eigenvalues" + Astr + "_particles"

folder_2d_space="../A2_space_solution/data_s_V" + pos_neg +"20/xL5.0_Nx200/"
folder_PT="../A2_perturbation_theory/data_s_V" + pos_neg +"20/"

# SPACE DATA
fname_2d= folder_2d_space + file_space
if os.path.exists(fname_2d) :
    xspace = np.loadtxt(fname_2d,unpack=True)
    s0sp=xspace[1,:].T

# PT1 DATA
fname_PT1= folder_PT + file0_PT + "_order1.dat"
if os.path.exists(fname_PT1) :
    xPT1 = np.loadtxt(fname_PT1,unpack=True)
    s0pt1=xPT1[1,:].T

# PT2 DATA
fname_PT2= folder_PT + file0_PT + "_order2.dat"
if os.path.exists(fname_PT2) :
    xPT2 = np.loadtxt(fname_PT2,unpack=True)
    s0pt2=xPT2[1,:].T

# PT3 DATA
fname_PT3= folder_PT + file0_PT + "_order3.dat"
if os.path.exists(fname_PT3) :
    xPT3 = np.loadtxt(fname_PT3,unpack=True)
    s0pt3=xPT3[1,:].T

# DIAGONALIZATION DATA
folder_diag="../Diagonalization/data_s_V/"
filediag="Hamiltonian_eigenvalues2_particles.dat"
fname_diag= folder_diag + filediag
if( pos_neg == "-" ) : icol=1
if( pos_neg == "+" ) : icol=2

c_full = plt.cm.Blues(np.linspace(1,0.3,nstate))

# NONINTERACTING BASELINE
for istate in range(nstate) :
    efree=np.power(A_num_part,2)/2 + istate
    s0range=np.linspace(0,10,10)
    ax.plot(s0range,efree*np.ones_like(s0range),":",color='#969696')

# NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/srange_data/"
    fileML_app="_FUNC_Tanh_V0" + pn + "2.00e+01.txt"
    file_ML="data_s_V_A0" + Astr + "_NHID" + HH + "_NLAY" + LL + "_NDET" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    if os.path.exists(fname_ML) :
        s_ml,e_ml,ee_ml = np.loadtxt(fname_ML,usecols=(0,1,2),unpack=True)
        iplot=1
        ax.fill_between(s_ml,e_ml-ee_ml,e_ml+ee_ml,color='#377eb8',alpha=0.8,zorder=3)
        ax.plot(s_ml,e_ml,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])


# HARTREE-FOCK SOLUTION
# HARTREE FOCK DATA
    folder_HF="../hartree_fock/data_s_V" + pos_neg + "20/xL6.0_Nx240/"
    fname_HF= folder_HF + file_space
    if os.path.exists(fname_HF) :
        print(fname_HF)
        iplot=3
        s_hf,e_hf = np.loadtxt(fname_HF,usecols=(1,2),unpack=True)
        ax.plot(s_hf,e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    if os.path.exists(fname_2d) :
        print(fname_2d)
        iplot=4
        ax.plot(s0sp,xspace[istate+2,:].T,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot],markersize=10)

    if os.path.exists(fname_diag) :
        iplot=0
        s_d,e_d = np.loadtxt(fname_diag,usecols=(0,icol),unpack=True)
        ax.plot(s_d,e_d,"--",color=linestyle_color[iplot],alpha=1,label="Diag",zorder=3)

    if os.path.exists(fname_PT1) :
        iplot=2
        ax.plot(s0pt1,xPT1[istate+2,:].T,"-", dashes=[6, 2, 1, 2, 1, 2],color=linestyle_color[iplot],alpha=0.5,label="PT1")

    if os.path.exists(fname_PT2) :
        ax.plot(s0pt2,xPT2[istate+2,:].T,"-.",color=linestyle_color[iplot],alpha=0.75,label="PT2")

    if os.path.exists(fname_PT3) :
        ax.plot(s0pt3,xPT3[istate+2,:].T,"--",color=linestyle_color[iplot],alpha=1,label="PT3")




if(pos_neg == "-") : ax.legend(frameon=False,handlelength=3.2,fontsize=16,handletextpad=0.25,ncol=2,loc="upper right")
if(pos_neg == "+") : ax.legend(frameon=False,handlelength=3.2,fontsize=16,handletextpad=0.25,ncol=2)
ax.set_xlabel("Range, $\sigma_0$")
ax.set_ylabel("Ground state energy, $E$")
ax.set_title("A=" + Astr + ", $V_0=" + pos_neg + "20$" )
ax.set_xlim([0,3])

ax.tick_params(which='both',direction='in',top=True,right=True)
ax.tick_params(which='major',length=8)
ax.tick_params(which='minor',length=4)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

file_figure="energy_srange_A" + Astr + "_V"  + pos_neg + "20.pdf"
print(file_figure)
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
