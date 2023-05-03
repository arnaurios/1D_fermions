# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES THE TWO-BODY PROBLEM IN 1D
# 2 PARTICLES IN A HO TRAP WITH A GAUSSIAN INTERACTION
##########################################################################
import sys
import numpy as np
from scipy import linalg as LA
import math
import scipy
import pandas as pd
from scipy import interpolate

import os

from harmonic_oscillator import *
##########################################################################
pi=math.pi
zi=complex(0.,1.)

eps_system=sys.float_info.epsilon
zero_low=eps_system*1000

# DEFINES THE VALUES OF INTERACTION STRENGTH AND INTERACTION RANGE
# THIS IS OPTION FOR FIXED S AND CHANGING V
nV=41
nS=1
V_strength=np.linspace(-20,20,nV)
S_range=np.linspace(0.5,0.5,nS)

# THIS IS OPTION FOR CHANGING S AND FIXED V
#nV=1
#nS=30
#V_strength=np.linspace(-20,-20,nV)
#S_range=(np.linspace(0.1,3.,nS))

print(V_strength)
print(S_range)

VV,ss=np.meshgrid(V_strength,S_range)

# DEFINES NUMBER OF EIGENVALUES THAT ARE COMPUTED
neig_rel=6
neig_CM=6

# REAL-SPACE MESH DIMENSION AND NUMBER POINTS
xL=5.
Nx=200
folder_numerics="xL" + str(xL) + "_Nx" + str(Nx)

######################################################
# CREATE DATA DIRECTORY IF THEY DO NOT EXIST
datafolder="data"
if( nV == 1 and nS > 1) :
    vstring="{0:+.0f}".format(V_strength[0])
    datafolder="data_s_V" + vstring

if( nS == 1 and nV > 1) :
    sstring="{0:.1f}".format(S_range[0])
    datafolder="data_s" + sstring + "_V"

print(datafolder)

if not os.path.exists(datafolder):
    os.makedirs(datafolder)

data_folder=datafolder + "/" + folder_numerics + "/"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

print(data_folder)

# GRID SPACING
delx=2*xL/Nx

# MESH IN X-SPACE - FROM -xL+del x UP TO +xL
xx=np.zeros(Nx)
xx=delx*(np.arange(Nx)-Nx/2.+1)
indx_origin=int(Nx/2-1)
[x1,x2]=np.meshgrid(xx,xx)
xcm=(x1+x2)/np.sqrt(2)
xrl=(x1-x2)/np.sqrt(2)

# SPACING IN MOMENTUM SPACE
delp=2.*pi/(2.*xL)

# MESH IN p SPACE
pp=np.zeros(Nx)
#pp=delp*(np.arange(Nx)-(Nx-1.)/2.)
#pp=delp*(np.arange(Nx)-Nx/2.)
#print(pp)
for i in range(0,Nx) :
    pp[i]=(i-Nx/2)*delp

# SECOND DERIVATIVE MATRIX
cder2=np.zeros((Nx,Nx),complex)
der2=np.zeros((Nx,Nx))
# LOOP OVER I AND J
for i, xi in enumerate( xx ) :
    for j, xj in enumerate( xx ) :
        cder2[i,j] = np.dot( np.exp( zi*(xj-xi)*pp ), np.power(pp,2) )

# ADD PHYSICAL FACTORS AND KEEP REAL PART ONLY FOR SECOND DERIVATIVE
der2=-np.real(cder2)*delx*delp/2./pi

# HARMONIC OSCILLATOR MATRIX IN REAL SPACE - DIAGONAL
V_HO=np.power(xx,2.)/2.
# HARMONIC OSCILLATOR EIGENVALUES FOR CENTER OF MASS
E_CM=np.arange(neig_CM)+0.5

# DECLARE SOME ARRAYS
en_rel=np.zeros(neig_rel)
wf_rel=np.zeros((Nx,neig_rel))

etot_mat=np.zeros((neig_CM,neig_rel))
wf_mat=np.zeros((Nx,Nx,neig_CM,neig_rel))

# DEFINE MATRICES AS A FUNCTION OF S AND V
energy=np.zeros((neig_CM*neig_rel,nV,nS))
n_OBDM=np.zeros((Nx+1,nV,nS))
A_num_sum=np.zeros((nV,nS))
rms_den=np.zeros((nV,nS))
#nat_orb=np.zeros((Nx+1,neig_CM*neig_rel,nS,nV))

# LOOP OVER INTERACTION RANGE
for iS,s in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
    for iV,V0 in enumerate( V_strength ) :
        V0string="{:.1f}".format(V0)

        print("\ns_range=" + str(s) + " V0=",V0string)

        # INTERACTION POTENTIAL IN COORDINATE SPACE
        V_Gauss=V0/np.sqrt(2.*pi)/s*np.exp( - np.power(xx,2.)/np.power(s,2.) );

        # INTERACTION Nx x Nx MATRIX - DIAGONAL IN R SPACE
        V=np.diag( V_HO + V_Gauss )

        # HAMILTONIAN Nx x Nx MATRIX
        H=-der2/2 + V

        # COMPUTE EIGENVALUES AND EIGENVECTORS OF RELATIVE COORDINATE MATRIX
        eivals,eivecs=LA.eigh(H,driver="evx")

        # SORT EIGENVALUES AND EIGENVECTORS
        isort=np.argsort(eivals)
        eivals=eivals[isort]
        eivecs=eivecs[isort]

        # ENERGY ORDERING COULD BE USED TO PICK ONLY ANTISYMMETRIC COMPONENTS
        # BUT WE DO IT EXPLICITLY BY PICKING STATE WITH WAVEFUNCTIONS THAT CANCEL
        # AT THE ORIGIN
        ieig_rel=0
        ieig=0
        while ieig_rel < neig_rel :

            # WAVEFUNCTION AT THE ORIGIN
            wf_at_origin=eivecs[indx_origin,ieig]
            if ( abs(wf_at_origin) < zero_low ) :
                sss=eivecs[indx_origin+2,ieig]
                sss=sss/np.abs(sss)
                wf_rel[:,ieig_rel] = sss*eivecs[:,ieig]/np.sqrt(delx);
                en_rel[ieig_rel] =eivals[ieig]
                ieig_rel=ieig_rel+1

            ieig=ieig+1
        # AT THIS STAGE, WE HAVE EIGENVALUES AND WAVEFUNCTIONS OF RELATIVE MOTION
        ########################################################################

        ########################################################################
        # ADD CM EIGENVALUES AND WAVEFUNCTIONS
        # OPEN LOOP OVER RELATIVE EIGENSTATES
        for i_erel,erl in enumerate( en_rel ) :
            www=wf_rel[:,i_erel]
            wfrl=np.zeros((Nx,Nx))
            # INTERPOLATE WAVEFUNCTION FROM 1D RELATIVE COORDINATE TO COORDINATES x1 AND x2
            for ix1,xx1 in enumerate( xx ) :
                for ix2,xx2 in enumerate( xx ) :
                    xrl=(xx2-xx1)/np.sqrt(2.)
                    # CUBIC NATURE OF INTERPOLATION RATHER CRITICAL FOR OCCUPATION NUMBER DETERMINATION
                    interp = interpolate.interp1d(xx, www, kind = "cubic",fill_value=0.,bounds_error=False)
                    #interp = interpolate.interp1d(xx, www, kind = "linear", fill_value=0.,bounds_error=False)
                    wfrl[ix1,ix2]=interp(xrl)

            # OPEN LOOP OVER CENTER OF MASS EIGENSTATES
            for i_eCM,eCM in enumerate( E_CM ) :
                etot_mat[i_eCM,:]=eCM + en_rel
                wfcm=wfho(i_eCM,xcm)

                # THIS IS THE TOTAL WAVEFUNCTION IN COORDINATES x1 AND x2
                wf_mat[:,:,i_eCM,i_erel]=wfcm*wfrl

        # SORT ENERGY EIGENVALUES AND EIGENVECTORS
        isort = (etot_mat).argsort(axis=None, kind='mergesort')
        j = np.unravel_index(isort, etot_mat.shape)
        etot=etot_mat[j]
        energy[:,iV,iS]=etot
        print("Energies")
        print( np.array2string(etot[0:9], formatter={'float_kind':'{0:.3f}'.format}) )

        wavefunctions=wf_mat[:,:,j[0],j[1]]
        ########################################################################

        ########################################################################
        # DENSITY MATRIX FOR GROUND STATE
        nstate=0
        denmat=-2*delx*np.matmul( wavefunctions[:,:,nstate],wavefunctions[:,:,nstate] )
        denmatr=np.r_[denmat , [denmat[1,:]] ]
        ddd=np.hstack( (denmat[:,1], 0) )
        denmatx=np.c_[denmatr, ddd ]

        # COMPUTE THE DENSITY
        density=np.diagonal(denmat).copy()

        # NATURAL ORBITALS AND OCCUPATIONS
        n_occ0,n_orb0=LA.eigh(delx*denmatx)
        nsort=np.argsort( -(n_occ0) )
        n_occupation=n_occ0[nsort]
        n_orbitals=np.squeeze( n_orb0[0:Nx,nsort]/np.sqrt(delx) )

        n_OBDM[:,iV,iS]=n_occupation
        # THIS WOULD COMPUTE NATURAL ORBITALS
        #nat_orb[:,:,iS,iV]=n_orbitals

        print("Occupation numbers")
        print( np.array2string(n_occupation[0:9], formatter={'float_kind':'{0:.3f}'.format}) )

        A_num_sum[iV,iS]=delx*np.sum( density )
        rms_den[iV,iS]= delx*np.dot( density,np.power(xx,2) )/ (delx*np.sum( density ))

        ########################################################################

        ########################################################################
        # OUTPUT CODE RESULTS - RANGE AND STRENGTH DEPENDENCE

        ########################################################################
        # SAVE ONE BODY DENSITY
        # DENSITY
        outputfile=data_folder + "density_2_particles_V0=" + V0string + ".dat"
        formatd=["%16.6E"] * 2
        data_to_write = np.array( [ xx[:],density[:] ] ).T
        header_density="# V0=" + V0string + ", s=" + str(s) + " \n#   x [ho units]    Density [ho units]"
        with open(outputfile,"w+") as file_id :
            np.savetxt(file_id,[header_density],fmt="%s")
            np.savetxt(file_id,data_to_write,fmt=formatd)

        # RELATIVE WAVEFUNCTIONS
        outputfile=data_folder + "wavefunction_rel_V0=" + V0string + ".dat"
        formatd=["%16.6E"] * 2
        data_to_write = np.array( [ xx[:],wf_rel[:,0] ] ).T
        header="# V0=" + V0string + ", s=" + str(s) + " \n#   x [ho units]    Relative wf [ho units]"
        with open(outputfile,"w+") as file_id :
            np.savetxt(file_id,[header],fmt="%s")
            np.savetxt(file_id,data_to_write,fmt=formatd)

        # PAIR DISTRIBUTION FUNCTIONS
        outputfile=data_folder + "pair_distribution_function_V0=" + V0string + ".dat"
        formatd=["%16.6E"] * 2
        data_to_write = np.array( [ xx[:],np.power(wf_rel[:,0],2) ] ).T
        header="# V0=" + V0string + ", s=" + str(s) + " \n#   x [ho units]    Relative wf [ho units]"
        with open(outputfile,"w+") as file_id :
            np.savetxt(file_id,[header],fmt="%s")
            np.savetxt(file_id,data_to_write,fmt=formatd)

        # WAVE FUNCTION
        formatr=["%16.6E"] * 4
        outputfile=data_folder + "2D_wavefunction_2_particles_V0=" + V0string + ".dat"
        data_to_write = np.array( [ x1[:,:],x2[:,:],wavefunctions[:,:,0] ] ).T
        header_denmat="# V0=" + str(V0) + ", s=" + str(s) + " \n#  x1  x2  wavefunction"
        with open(outputfile,"w+") as file_id :
            np.savetxt(file_id,[header_denmat],fmt="%s")
            for line in data_to_write:
                np.savetxt(file_id,line,fmt=["%16.6E","%16.6E","%16.6E"],header="  ")

        # DENSITY MATRIX
        formatr=["%16.6E"] * 4
        outputfile=data_folder + "denmat_2_particles_V0=" + V0string + ".dat"
        data_to_write = np.array( [ x1[:,:],x2[:,:],denmat[:,:] ] ).T
        header="# V0=" + str(V0) + ", s=" + str(s) + " \n#  x1  x2  denmat"
        with open(outputfile,"w+") as file_id :
            np.savetxt(file_id,[header],fmt="%s")
            for line in data_to_write:
                np.savetxt(file_id,line,fmt=["%16.6E","%16.6E","%16.6E"],header="  ")

        ########################################################################

        ########################################################################

########################################################################
# OUTPUTS CODE - AS A FUNCTION OF RANGE AND STRENGTH
# WRITING OUTPUT IN FILE

# ENERGIES
outputfile=data_folder + "Hamiltonian_eigenvalues2_particles.dat"
header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Energy (n=0) [ho]" + "  Energy (n=1) [ho] ..."
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")

# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + energy[0:14,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")

# OCCUPATION NUMBERS
outputfile=data_folder + "OBDM_eigenvalues_2_particles.dat"
header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Occupation (n=0)" + "  Occupation (n=1) ..."
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")
# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + n_OBDM[0:14,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")

# NORMALIZATION AND RMS
outputfile=data_folder + "x2_2_particles.dat"
header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "<x^2> [ho units]" + "  Norm denmat []"
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")
# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss,rms_den[iV,iS],A_num_sum[iV,iS] ] )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")
