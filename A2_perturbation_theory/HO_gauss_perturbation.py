# PERTURBATION THEROY CALCULATION - FIRST AND SECOND ORDER RESULTS FOR
# GAUSSIAN+HO
# n_target=state that is target in relative coordinates
# V0 and s can be vectors - output matrix is E_PTx(V0,s)
import numpy as np
import math

# WAVE FUNCTIONS OF THE 1D HARMONIC OSCILLATOR
def E_Perturbation(V_inp,s_inp,n_target) :

    pi=math.pi

    # EARL PAPER - J. Chem. Educ. 2008, 85, 3, 453. https://doi.org/10.1021/ed085p453
    # THIS SHEET REPRODUCES THE DATA AS PRESENTED IN THE
    # APPENDIX AND TABLE I

    # ISSUE WHEN DIM S neq DIM V0 OR 1!!!
    bb=V_inp/np.sqrt(2.*pi)/s_inp
    cc=1/np.power(s_inp,2)

    # TARGETED STATE
    # H_ii FOR TARGETED STATE
    H0target=0.5+n_target

    N_additional_states=9
    nmax=20
    N_additional_states=int( np.floor( (nmax-n_target)/2. ) )
    #print( N_additional_states )
    # BUILD VECTOR AND MATRICES FOR 2ND AND 3RD ORDER RESULTS

    # LOOP OVER INTERACTION STRENGTH
    # FIRST ORDER PT RESULT
    V00=bb*V_HO_gaussian(n_target,n_target,cc)
    E_PT1=H0target+V00

    # SECOND ORDER PT RESULT - DEFINE MATRICES
    V0=np.zeros([2*N_additional_states,1]);
    VH=np.zeros([2*N_additional_states,1]);
    VV=np.zeros([2*N_additional_states,2*N_additional_states]);

    # BUILD PT MATRICES
    ii=0
    for nn in range(n_target+2,n_target+2*N_additional_states +1) :
        Vij=bb*V_HO_gaussian(n_target,nn,cc)
        jj=0
        for mm in range(n_target+2,n_target+2*N_additional_states +1) :
            VV[jj,ii]=bb*V_HO_gaussian(mm,nn,cc)
            jj=jj+1

        V0[ii]=Vij
        Hii=(0.5+nn)
        VH[ii]=Vij/(H0target-Hii)
        ii=ii+1

    E_second=np.matmul(VH.T,V0)
    E_PT2=E_PT1+E_second

    E_third_1=-V00 * np.matmul(VH.T,VH)
    #print("31",E_third_1)
    E_third_2= np.matmul( VH.T, np.matmul(VV,VH) )
    #print("32",E_third_2)
    E_PT3=E_PT2+E_third_1+E_third_2

    #print(E_PT1)
    #print(E_PT2)
    #print(E_PT3)
    return E_PT1,E_PT2,E_PT3


    # HO 1D WAVEFUNCTION
    # OVERLAP OF GAUSSIAN WITH HARMONIC OSCILLATOR WAVEFUNCTION
    # EARL - The Harmonic Oscillator with a Gaussian Perturbation: Evaluation of the Integrals and Example Applications
def V_HO_gaussian(n,m,g) :

    if n > 20 :
        print("n in wavefunction beyond accuracy: exiting code")
        exit()

    V=0
    if( n%2 == m%2 ) :
        nfac=math.factorial(n)
        mfac=math.factorial(m)

        xnorm=np.sqrt(nfac*mfac/np.power(2,n+m)/(1.+g));

        lmin=n%2
        lmax=min(n,m)

        # MADE THIS WAY TO FORCE LOOP EVALUATION WITH L
        number_l=int( (lmax-lmin)/2+1 )
        lvals = np.linspace(lmin,lmax,number_l)
        for il,l in enumerate( lvals ) :
            lfac=math.factorial( int(l) )

            k=(m-l)/2
            j=(n-l)/2

            kfac=math.factorial( int(k) )
            jfac=math.factorial( int(j) )

            Vjkl=np.power(g,(j+k))/np.power((1+g),(j+k+l))*np.power(-1,j+k)*np.power(2,l)/jfac/kfac/lfac;

            V=V+Vjkl
        V=V*xnorm
    return V
