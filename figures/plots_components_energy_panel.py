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

line_label = [r'Total, $\langle H \rangle$',r'Kinetic, $\langle K \rangle$',r'Potential, $\langle U \rangle$',r'Interaction, $\langle V \rangle$']
linestyle_color = ['#377eb8','#4daf4a','#984ea3','#e41a1c']
linestyle_width = [3,3,3,3]
linestyle_order = [1,2,3,4]
linestyle_dashes = ['-','--','-.',':']

ymin=[- 8,-20,-40,-60,-80]
ymax=[  5, 12, 25, 40, 50]
minticks=[5,5,4,4,5]

# NUMBER OF PARTICLES
A=[2,3,4,5,6]

alphabet=["(a)","(b)","(c)","(d)","(e)"]

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

fileML_app="_FUNC_Tanh_SIG0_5.00e-01.npz"


fig,ax = plt.subplots(nrows=num_plots,ncols=1,figsize=(7,3.5*num_plots),sharex=True)

for iy,A_num_part in enumerate(A) :
    Astr=str(A_num_part)
    AA=Astr.zfill(2)
    print(Astr,AA)

    # NONINTERACTING BASELINE
    efree=np.power(A_num_part,2)/2

    # NETWORK DATA SHOULD GO HERE
    folder_ML="../neural_network/Separate_Energy/"
    file_ML="SEP_A" + AA + "_NHID_" + HH + "_NLAY" + LL + "_NDET" + DD + fileML_app
    fname_ML= folder_ML + file_ML
    print(fname_ML)    
    if os.path.exists(fname_ML) :     
        print(fname_ML)    
        data = np.load(fname_ML) #load the PHYS.npz file
        total = data['total']
        kinetic = data['kinetic']
        potential = data['potential']
        interaction = data['interaction']

        iplot=0
        ax[iy].plot(total[:,0], total[:,1],linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
        iplot=1 
        ax[iy].plot(kinetic[:,0], kinetic[:,1],linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
        iplot=2 
        ax[iy].plot(potential[:,0], potential[:,1],linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])
        iplot=3
        ax[iy].plot(interaction[:,0], interaction[:,1],linestyle_dashes[iplot],color=linestyle_color[iplot],label=line_label[iplot],zorder=linestyle_order[iplot],linewidth=linestyle_width[iplot])

        eeee=np.ones_like(interaction[:,0],)*efree
        ax[iy].plot(interaction[:,0],eeee,":",color='#969696',zorder=0,linewidth=2)


    ax[iy].set_xlim([-20,20])
    ax[iy].set_ylim([ymin[iy],ymax[iy]])

    ax[iy].tick_params(which='both',direction='in',top=True,right=True)
    ax[iy].tick_params(which='major',length=8)
    ax[iy].tick_params(which='minor',length=4)
    ax[iy].xaxis.set_ticks(np.arange(-20, 20.00001, 5))
    ax[iy].xaxis.set_minor_locator(AutoMinorLocator(5))

    ax[iy].yaxis.set_minor_locator(AutoMinorLocator(minticks[iy]))


    ax[iy].text(0.76, 0.87, "A=" + Astr + " " + alphabet[iy],transform=ax[iy].transAxes, fontsize=22)

ax[0].legend(frameon=False,handlelength=2.5,fontsize=18,handletextpad=1)

ax[2].set_ylabel("Energy components")

ax[num_plots-1].set_xlabel("Strength, $V_0$")
ax[0].set_title("$\sigma_0=0.5$")

file_figure="energy_components_panel.pdf"
print(file_figure)
#plt.show()
plt.tight_layout()
plt.savefig(file_figure, bbox_inches='tight')
plt.close(fig)
