# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.ticker import AutoMinorLocator

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
linestyle_order = [2,3,0,1,4]
linestyle_dashes = ['--','-','','-.','.']

#pos_neg=input("Pos or neg value + -:")
print("Pos or neg value + -:")
pos_neg = sys.argv[1]
pn=pos_neg 
if(pos_neg == "+") : pn=""


# NUMBER OF PARTICLES
alphabet=["(a)","(b)","(c)","(d)","(e)"]
A=[2,3,4,5,6]
#A=[3,4,5,6]
num_plots=len(A)
print(num_plots)

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

fig,ax = plt.subplots(nrows=num_plots,ncols=1,figsize=(7,3.5*num_plots),sharex=True)


for iy,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    AA=Astr.zfill(2)
    print(Astr,AA)

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
    istate=0

    iplot=2
    fname_PT1= folder_PT + file0_PT + "_order1.dat"
    if os.path.exists(fname_PT1) :
        xPT1 = np.loadtxt(fname_PT1,unpack=True)
        s0pt1=xPT1[1,:].T
        ax[iy].plot(s0pt1,xPT1[istate+2,:].T,"-", dashes=[6, 2, 1, 2, 1, 2],color=linestyle_color[iplot],alpha=0.5,label="PT1")

    # PT2 DATA
    fname_PT2= folder_PT + file0_PT + "_order2.dat"
    if os.path.exists(fname_PT2) :
        xPT2 = np.loadtxt(fname_PT2,unpack=True)
        s0pt2=xPT2[1,:].T
        ax[iy].plot(s0pt2,xPT2[istate+2,:].T,"-.",color=linestyle_color[iplot],alpha=0.75,label="PT2")

    # PT3 DATA
    fname_PT3= folder_PT + file0_PT + "_order3.dat"
    if os.path.exists(fname_PT3) :
        xPT3 = np.loadtxt(fname_PT3,unpack=True)
        s0pt3=xPT3[1,:].T
        ax[iy].plot(s0pt3,xPT3[istate+2,:].T,"--",color=linestyle_color[iplot],alpha=1,label="PT3")

# NONINTERACTING BASELINE
    efree=np.power(A_num_part,2)/2 #+ istate
    s0range=np.linspace(0,10,10)
    ax[iy].plot(s0range,efree*np.ones_like(s0range),":",color='#969696')

    if os.path.exists(fname_2d) :
        iplot=4
        ax[iy].plot(s0sp,xspace[istate+2,:].T,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
        #ax[iy].plot(s0sp,xspace[istate+2,:].T,".",color="#ff7f00")

# HARTREE FOCK DATA
    folder_HF="../hartree_fock/data_s_V" + pos_neg + "20/xL6.0_Nx240/"
    fname_HF= folder_HF + file_space
    if os.path.exists(fname_HF) :
        s_hf,e_hf = np.loadtxt(fname_HF,usecols=(1,2),unpack=True)
        iplot=3
        ax[iy].plot(s_hf,e_hf,linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

# NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/srange_data/"
    fileML_app="_FUNC_Tanh_V0" + pn + "2.00e+01.txt"
    file_ML="data_s_V_A0" + Astr + "_NHID" + HH + "_NLAY" + LL + "_NDET" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    if os.path.exists(fname_ML) :
        s_ml,e_ml,ee_ml = np.loadtxt(fname_ML,usecols=(0,1,2),unpack=True)
        ax[iy].fill_between(s_ml,e_ml-ee_ml,e_ml+ee_ml,color='#377eb8',alpha=0.8,zorder=3)
        iplot=1
        ax[iy].plot(s_ml,e_ml,color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

    ax[1].set_ylabel("Ground state energy, $E$")
    ax[iy].set_xlim([0,3])
    
    ax[iy].tick_params(which='both',direction='in',top=True,right=True)
    ax[iy].tick_params(which='major',length=8)
    ax[iy].tick_params(which='minor',length=4)
    ax[iy].xaxis.set_ticks(np.arange(0, 3.00000001, 1))
    ax[iy].xaxis.set_minor_locator(AutoMinorLocator(4))

    ax[iy].text(0.03, 0.87, alphabet[iy]+"  A=" + Astr,transform=ax[iy].transAxes, fontsize=22)

ax[0].legend(frameon=False,handlelength=2,fontsize=16,handletextpad=1,ncols=2)
ax[num_plots-1].set_xlabel("Range, $\sigma_0$")
ax[0].set_title("$V_0=" + pos_neg +  "20$")

file_figure="energy_srange_panel_V"  + pos_neg + "20.pdf"
print(file_figure)
#plt.tight_layout()
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
