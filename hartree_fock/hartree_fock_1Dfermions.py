# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES THE TWO-BODY PROBLEM IN 1D
# 2 PARTICLES IN A HO TRAP WITH A GAUSSIAN INTERACTION
##########################################################################
import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import scipy
#import scipy.fft
import pandas as pd
from scipy import interpolate

import os

from harmonic_oscillator import *
##########################################################################
pi=math.pi
zi=complex(0.,1.)

eps_system=sys.float_info.epsilon
zero_low=eps_system*1000

# NUMBER OF PARTICLES
A_num_part=6
Astr=str(A_num_part)
print(Astr)

# DEFINES THE VALUES OF INTERACTION STRENGTH AND INTERACTION RANGE
#nV=41
nV=3
nS=1
V_strength=np.linspace(-20,20,nV)
S_range=np.linspace(0.5,0.5,nS)
VV,ss=np.meshgrid(V_strength,S_range)

#nV=1
#nS=30
#S_range=np.linspace(0.1,3,nS)
#V_strength=np.linspace(-20,-20,nV)

# REAL-SPACE MESH DIMENSION AND NUMBER POINTS
xL=6.
Nx=240
folder_numerics="xL" + str(xL) + "_Nx" + str(Nx)

######################################################
# CREATE DATA/PLOT DIRECTORY IF THEY DO NOT EXIST
# PREPARE DATA AND PLOT FOLDERS
datafolder="data"
plotfolder="plots"
if not os.path.exists(datafolder):
    os.makedirs(datafolder)

if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

if not os.path.exists("data/" + folder_numerics):
    os.makedirs("data/" + folder_numerics)
data_folder="data/" + folder_numerics + "/"

if not os.path.exists("plots/" + folder_numerics):
    os.makedirs("plots/" + folder_numerics)
plot_folder="plots/" + folder_numerics + "/"
######################################################

# GRID SPACING
delx=2*xL/Nx

# MESH IN X-SPACE - FROM -xL+del x UP TO +xL
xx=np.zeros(Nx)
#xx=np.arange(-xL+delx,xL,delx);
#print(xx)
xx=delx*(np.arange(Nx)-Nx/2.)
indx_origin=int(Nx/2)
[x1,x2]=np.meshgrid(xx,xx)

# SPACING IN MOMENTUM SPACE
delp=2.*pi/(2.*xL)

# MESH IN p SPACE
pp=np.zeros(Nx)
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
        #cder2[i,j] = np.dot( np.cos( (xj-xi)*pp )+zi*np.sin( (xj-xi)*pp ), np.power(pp,2) )

# ADD PHYSICAL FACTORS AND KEEP REAL PART ONLY FOR SECOND DERIVATIVE
der2=-np.real(cder2)*delx*delp/2./pi
kin_mat=-der2/2. # COULD ADD HBAR2/M HERE IF OTHER UNITS USED


# SECOND DERIVATIVE MATRIX - EXPERIMENTAL FEATURE TO BE CHECKED
#cder2=np.zeros((Nx,Nx),complex)
#der2=np.zeros((Nx,Nx))
#np.fill_diagonal( cder2,np.power(pp,2) )
#print(cder2[0:4,0:4])
#cder2=scipy.fft.fft2(cder2.T)
# ADD PHYSICAL FACTORS AND KEEP REAL PART ONLY FOR SECOND DERIVATIVE
#der2=-np.real(cder2)*delx*delp/2./pi
#kin_mat=-der2/2. # COULD ADD HBAR2/M HERE IF OTHER UNITS USED


# HARMONIC OSCILLATOR MATRIX IN REAL SPACE - DIAGONAL
U_HO=np.power(xx,2.)/2.

# HARTREE-FOCK POTENTIAL
accu=1e-9
itermax=20000
pfac=1./np.abs( np.amax(kin_mat) )
ffac=0.4

# PREPARE ARRAYS
hf_den=np.zeros(Nx)
denf=np.zeros(Nx)
Uex=np.zeros(Nx)
hf_den_mat=np.zeros((Nx,Nx))
H = np.zeros((Nx,Nx))

# INITIALIZE ARRAYS
# NOTE Nmax IS USED AS MAXIMUM IN PYTHON RANGE ARRAYS, SO IT IS ACTUALLY A-1 IN MATHS TERMS
Nmax=A_num_part
wfy=np.zeros((Nx,Nmax))
spe=np.zeros(Nmax)
for ieig in range(Nmax) :
    wfy[:,ieig] = wfho(ieig,xx)

# DEFINE MATRICES AS A FUNCTION OF S AND V
energy=np.zeros((5,nV,nS))
A_num_sum=np.zeros((nV,nS))
rms_den=np.zeros((nV,nS))

#H0 = np.zeros((Nx,Nx))
#H0 = kin_mat.copy()
#np.fill_diagonal(H0,H0.diagonal() + U_HO)
# COMPUTE EIGENVALUES AND EIGENVECTORS OF RELATIVE COORDINATE MATRIX
#eivals,eivecs=LA.eigh(H0)
#
# SORT EIGENVALUES AND EIGENVECTORS
#isort=np.argsort(eivals)
#eivals=eivals[isort]
#eivecs=eivecs[isort]

#for ieig in range(Nmax) :
#    wwf = eivecs[:,ieig]/np.sqrt(delx)
#    wfy[:,ieig] = wwf
header_screen="# ITER".ljust(8)+"NUM_PART".ljust(14) + "X_CM".ljust(13) + "EHF".ljust(13) + "EHF2".ljust(13) + "EKIN".ljust(13) + "EPOT".ljust(13) + "ESUM".ljust(13) + "DIFFS"

# LOOP OVER INTERACTION RANGE
for iS,s in enumerate( S_range ) :
    s_string="{:.1f}".format(s)
# LOOP OVER INTERACTION STRENGTH
    for iV,V0 in enumerate( V_strength ) :
        V0string="{:.1f}".format(V0)
        print("\ns_range=" + s_string + " V0=",V0string)
        print(header_screen)

        # INTERACTION POTENTIAL MATRIX
        Vint=V0/np.sqrt(2.*pi)/s*np.exp( -np.power(x1-x2,2)/2./np.power(s,2) )

        # START HARTREE-FOCK ITERATION PROCEDURE
        iter=0
        diffs=10.
        while ( diffs > accu and iter<itermax ) :
            iter=iter+1
            # ... PREPARE DENSITY AND DENSITY MATRIX FROM ORBITALS
            hf_den=0.
            hf_den_mat=0.
            for ieig in range(Nmax) :
                hf_den=hf_den+ np.power( abs( wfy[:,ieig] ), 2)
                hf_den_mat=hf_den_mat + np.outer( wfy[:,ieig], wfy[:,ieig] )

            # ... COMPUTE MEAN-FIELD
            denf=hf_den
            # DIRECT TERM
            Udir = delx*np.matmul(Vint,denf)
            # EXCHANGE TERM
            Umf_mat = -delx*Vint*hf_den_mat

            Uexc = np.diagonal(Umf_mat).copy()
            #print(Uexc[0:5])
            # ADD ALL MEAN-FIELD TERMS TOGETHER
            np.fill_diagonal( Umf_mat, Umf_mat.diagonal() + Udir + U_HO)
            # MEAN-FIELD ALONG DIAGONAL
            Umf = np.diagonal(Umf_mat).copy()

            # HAMILTONIAN Nx x Nx MATRIX
            H=kin_mat + Umf_mat

            #cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            # ... USE RAYLEIGH APPROXIMATION TO FIND APPROXIMATE EIGENVALUES
            # ... LOOP OVER EIGENVALUES
            diffs=0.
            ekin0hf=0.
            epot0hf=0.
            for ieig in range(Nmax) :
            # ... GUESS EIGENVALUE WITH RAYLEIGH METHOD
                wf0=wfy[:,ieig]
                wff=np.matmul( H, wf0 )
                spe[ieig]=np.real( np.dot( wf0 , wff) )*delx

            # GRADIENT ITERATION
                wfb=wf0 - wff*pfac
                norm=math.fsum( np.power( np.abs(wfb),2 ))*delx
                wff=wfb/np.sqrt(norm)
                #print(wff[99])

            # ... ORTHOGONALIZATION
                wfo=0.
                for jeig in range(0,ieig) :
                    wfo = wfo + wfy[:,jeig]*np.dot( wfy[:,jeig],wff )*delx
                wff=wff-wfo
                norm=math.fsum( np.power( np.abs(wff),2 ))*delx
                wff=wff/norm

                # ... vec_k+1 = (A-mI)^-1 * vec_k
                diffs=diffs + np.amax( abs(wf0 - wff) )
                wfy[:,ieig] = ffac*wff + (1.-ffac)*wf0

                if( ieig <= Nmax ) :
                    ekin0hf=ekin0hf + np.real( np.dot( wfy[:,ieig], np.matmul( kin_mat,wfy[:,ieig])) )*delx
                    epot0hf=epot0hf + np.real( np.dot( wfy[:,ieig], np.matmul( Umf_mat,wfy[:,ieig])) )*delx

            esum0hf=np.sum(spe[0:Nmax])

            # FIRST HARTREE-FOCK DETERMINATION USING KINETIC ENERGY
            xa=np.sum( hf_den )*delx
            x2_av=delx*np.dot( hf_den,np.power(xx,2) )/ (delx*np.sum( hf_den ))

            # ENERGY FROM GMK SUMRULE
            eho=np.sum( hf_den*U_HO ) *delx/2.
            enerhfp=(esum0hf+ekin0hf)/2.+eho

            # ENERGY FROM INTEGRAL (DF)
            epothf=epot0hf-2.*eho
            enerhf=esum0hf-epothf/2.

            # PRINT DATA TO SCREEN
            formatd=["%12.6E"] * 8
            formatd=["%4i"] + formatd
            if iter%50 == 0 :
                ddd=np.vstack([iter,xa,x2_av,enerhf,enerhfp,ekin0hf, epot0hf,esum0hf,diffs])
                data_to_write =np.column_stack( ddd )
                np.savetxt(sys.stdout,ddd.T,fmt=formatd)

        # ITERATION LOOP IS OVER - PRINT FINAL RESULTS
        energy[0,iV,iS]=enerhf
        energy[1,iV,iS]=enerhfp
        energy[2,iV,iS]=ekin0hf
        energy[3,iV,iS]=eho
        energy[4,iV,iS]=epothf

        print(V0,spe)

        A_num_sum[iV,iS]=xa
        rms_den[iV,iS]= x2_av

        # DENSITY
        if( nS == 1) :
            outputfiled=data_folder + "density_" + Astr +"_particles_V0=" + V0string + ".dat"
        elif( nV==1 ) :
            outputfiled=data_folder + "density_" + Astr +"_particles_s=" + s_string + ".dat"
        else :
            outputfiled=data_folder + "density_" + Astr +"_particles_V0=" + V0string + "_s=" + s_string + ".dat"

        if os.path.exists(outputfiled) :
            os.remove(outputfiled)

        formatd=["%16.6E"] * 2
        data_to_write = np.array( [ xx[:],denf[:] ] ).T
        header_density="\n\n# V0=" + V0string + ", s=" + str(s) + " \n#   x [ho units]    Density [ho units]"
        with open(outputfiled,"a") as file_id :
            np.savetxt(file_id,[header_density],fmt="%s")
            np.savetxt(file_id,data_to_write,fmt=formatd)

        # MEAN FIELD
        if( nS == 1) :
            outputfilem=data_folder + "meanfield_" + Astr +"_particles_V0=" + V0string + ".dat"
        elif( nV==1 ) :
            outputfilem=data_folder + "meanfield_" + Astr +"_particles_s=" + s_string + ".dat"
        else :
            outputfilem=data_folder + "meanfield_" + Astr +"_particles_V0=" + V0string + "_s=" + s_string + ".dat"

        if os.path.exists(outputfilem) :
            os.remove(outputfilem)
        formatd=["%16.6E"] * 5
        data_to_write = np.array( [ xx[:],Umf[:],U_HO[:],Udir[:],Uexc[:] ] ).T
        header_density="\n\n# V0=" + V0string + ", s=" + str(s) + " \n#   x [ho units]    Meanfield: Umf, UHO, Udir, Uexc [ho units]"
        with open(outputfilem,"a") as file_id :
            np.savetxt(file_id,[header_density],fmt="%s")
            np.savetxt(file_id,data_to_write,fmt=formatd)

        # DENSITY MATRIX
        if( nS == 1) :
            outputfiledm=data_folder + "denmat_" + Astr +"_particles_V0=" + V0string + ".dat"
        elif( nV==1 ) :
            outputfiledm=data_folder + "denmat_" + Astr +"_particles_s=" + s_string + ".dat"
        else :
            outputfiledm=data_folder + "denmat_" + Astr +"_particles_V0=" + V0string + "_s=" + s_string + ".dat"

        if os.path.exists(outputfiledm) :
            os.remove(outputfiledm)

        formatr=["%16.6E"] * 4
        data_to_write = np.array( [ x1[:,:],x2[:,:],hf_den_mat[:,:] ] ).T
        header="# V0=" + str(V0) + ", s=" + str(s) + " \n#  x1  x2  denmat"
        with open(outputfiledm,"w+") as file_id :
            np.savetxt(file_id,[header],fmt="%s")
            for line in data_to_write:
                np.savetxt(file_id,line,fmt=["%16.6E","%16.6E","%16.6E"],header="  ")

        ##############################################################################
        # PLOT DENMAT
        # DENSITY
        if( nS == 1) :
            plot_filedd=plot_folder + "density_" + Astr + "_particles_V0=" + V0string + ".pdf"
        elif( nV==1 ) :
            plot_filedd=plot_folder + "density_" + Astr + "_particles_s=" + s_string + ".pdf"
        else :
            plot_filedd=plot_folder + "density_" + Astr + "_particles_V0=" + V0string + "_s=" + s_string + ".pdf"

        if os.path.exists( plot_filedd ) :
            os.remove(plot_filedd)

        plt.xlabel("Distance, x [ho units]")
        plt.ylabel("Density, $n(r)$")
        plt.title("Density, A=" + Astr + ", $V_0=$" + V0string + ", $s=$" + str(s) )
        plt.plot(xx,denf)
        plt.savefig(plot_filedd)
        plt.close()


        if( nS == 1) :
            plot_filedm=plot_folder + "denmat_" + Astr + "_particles_V0=" + V0string + ".pdf"
        elif( nV==1 ) :
            plot_filedm=plot_folder + "denmat_" + Astr + "_particles_s=" + s_string + ".pdf"
        else :
            plot_filedm=plot_folder + "denmat_" + Astr + "_particles_V0=" + V0string + "_s=" + s_string + ".pdf"

        if os.path.exists( plot_filedm ) :
            os.remove(plot_filedm)

        fig, ax = plt.subplots()
        fcont=ax.contourf(xx,xx,hf_den_mat,cmap='coolwarm')
        ax.contour(xx,xx,hf_den_mat, colors='k')
        ax.axis("square")
        fig.colorbar(fcont,ax=ax)
        ax.set_xlabel("Position, $x_1$ [ho units]")
        ax.set_ylabel("Position, $x_2$ [ho units]")
        ax.set_title("Density matrix, A=" + Astr + ", $V_0=$" + V0string + ", $s=$" + str(s) )
        ax.set_xlim([-6,6])
        ax.set_ylim([-6,6])
        plt.savefig(plot_filedm)
        #plt.show()
        plt.close(fig)

        print()


# ENERGIES
outputfile=data_folder + "Hamiltonian_eigenvalues" + Astr + "_particles.dat"
header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "Energy (HF1) [ho]" + "  Energy (HF2)".ljust(20) + "EKIN".ljust(12) + "EHO".ljust(16) + "EPOT"
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")

# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss ] + energy[:,iV,iS].tolist() )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")

# NORMALIZATION AND RMS
outputfile=data_folder + "x2_" + Astr + "_particles.dat"
header="#".ljust(6) + "V0".ljust(12) + "s range".ljust(14) + "<x^2> [ho units]" + "  Norm denmat []"
with open(outputfile,"w+") as file_id :
    np.savetxt(file_id,[header],fmt="%s")
# LOOP OVER INTERACTION RANGE
    for iS,ss in enumerate( S_range ) :
# LOOP OVER INTERACTION STRENGTH
        for iV,vv in enumerate( V_strength ) :
            data_to_write =np.column_stack( [ vv,ss,rms_den[iV,iS],A_num_sum[iV,iS] ] )
            np.savetxt(file_id,data_to_write,fmt="%14.6E")
