# coding: utf-8
########################################################################
# GIVES ENERGY VALUES AT 1st, 2nd AND 3rd ORDER IN PERTURBATION THEORY
# FOR GROUND STATE OF GAUSSIAN INTERACTION IN A HARMONIC OSCILLATOR
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import os

from HO_gauss_perturbation import *

# DEFINES THE VALUES OF INTERACTION STRENGTH AND INTERACTION RANGE
nV=101
nS=1
V_strength=np.linspace(-20,20,nV)
S_range=np.linspace(0.5,0.5,nS)

nV=1
nS=60
V_strength=np.linspace(-20,-20,nV)
S_range=np.linspace(0.05,3,nS)

######################################################
# CREATE DATA DIRECTORY IF THEY DO NOT EXIST
datafolder="data"
if( nV == 1) :
    vstring="{0:+.0f}".format(V_strength[0])
    datafolder="data_s_V" + vstring + "/"

if( nS == 1) :
    sstring="{0:.1f}".format(S_range[0])
    datafolder="data_s" + sstring + "_V" + "/"

if not os.path.exists(datafolder):
    os.makedirs(datafolder)

# DEFINES NUMBER OF EIGENVALUES THAT ARE COMPUTED
neig_rel=10
neig_CM=10

# HARMONIC OSCILLATOR EIGENVALUES FOR CENTER OF MASS
E_CM=np.arange(neig_CM)+0.5

energy1=np.zeros((neig_CM*neig_rel,nV,nS))
energy2=np.zeros((neig_CM*neig_rel,nV,nS))
energy3=np.zeros((neig_CM*neig_rel,nV,nS))

# LOOP OVER INTERACTION RANGE
for iS,s in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
    for iV,V0 in enumerate( V_strength ) :
        eivals1=np.zeros(neig_rel)
        eivals2=np.zeros(neig_rel)
        eivals3=np.zeros(neig_rel)

        etot_mat1=np.zeros((neig_CM,neig_rel))
        etot_mat2=np.zeros((neig_CM,neig_rel))
        etot_mat3=np.zeros((neig_CM,neig_rel))

        V0string="{:.1f}".format(V0)

        print("\ns_range=" + str(s) + " V0=",V0string)
        nvals = np.arange(1,neig_rel*2,2 )
        for irel,nrel in enumerate(nvals) :
            #print(irel,nrel)
            eivals1[irel],eivals2[irel],eivals3[irel]=E_Perturbation(V0,s,nrel)

        # SORT EIGENVALUES AND EIGENVECTORS
        eivals1=np.sort( eivals1 )
        eivals2=np.sort( eivals2 )
        eivals3=np.sort( eivals3 )

        ########################################################################
        # AT THIS STAGE, WE HAVE EIGENVALUES AND WAVEFUNCTIONS OF RELATIVE MOTION
        ########################################################################
        # ADD CM EIGENVALUES AND WAVEFUNCTIONS
        # OPEN LOOP OVER RELATIVE EIGENSTATES
        for i_eCM,eCM in enumerate( E_CM ) :
            etot_mat1[i_eCM,:]=eCM + eivals1[:]
            etot_mat2[i_eCM,:]=eCM + eivals2[:]
            etot_mat3[i_eCM,:]=eCM + eivals3[:]

        # SORT ENERGY EIGENVALUES AND EIGENVECTORS
        #print(etot_mat1)
        isort = (etot_mat1).argsort(axis=None, kind='mergesort')
        j = np.unravel_index(isort, etot_mat1.shape)
        etot=etot_mat1[j]
        #print(etot)
        energy1[:,iV,iS]=etot

        print("Energies 1st Order")
        print( np.array2string(etot[0:9], formatter={'float_kind':'{0:.3f}'.format}) )

        isort = (etot_mat2).argsort(axis=None, kind='mergesort')
        j = np.unravel_index(isort, etot_mat2.shape)
        etot=etot_mat2[j]
        energy2[:,iV,iS]=etot_mat2[j]
        print("Energies 2nd Order")
        print( np.array2string(etot[0:9], formatter={'float_kind':'{0:.3f}'.format}) )

        isort = (etot_mat3).argsort(axis=None, kind='mergesort')
        j = np.unravel_index(isort, etot_mat3.shape)
        etot=etot_mat3[j]
        energy3[:,iV,iS]=etot_mat3[j]
        print("Energies 3rd Order")
        print( np.array2string(etot[0:9], formatter={'float_kind':'{0:.3f}'.format}) )

# ENERGIES - ORDER 1
outputfile=datafolder + "Hamiltonian_eigenvalues2_particles_order1.dat"
if os.path.exists(outputfile):
    os.remove(outputfile)

header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Energy (n=0) [ho]" + "  Energy (n=1) [ho] ..."
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")

# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + energy1[0:14,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")

# ENERGIES - ORDER 2
outputfile=datafolder + "Hamiltonian_eigenvalues2_particles_order2.dat"
if os.path.exists(outputfile):
    os.remove(outputfile)

header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Energy (n=0) [ho]" + "  Energy (n=1) [ho] ..."
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")

# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + energy2[0:14,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")

# ENERGIES - ORDER 3
outputfile=datafolder + "Hamiltonian_eigenvalues2_particles_order3.dat"
if os.path.exists(outputfile):
    os.remove(outputfile)

header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Energy (n=0) [ho]" + "  Energy (n=1) [ho] ..."
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")

# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + energy3[0:14,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")
