import numpy as np
import math

# WAVE FUNCTIONS OF THE 1D HARMONIC OSCILLATOR
def wfho(n,x) :

    if (n > 20) :
        print("N value of Harmonic Oscillator Wavefunction too large: exiting code")
        exit()

    nfac=math.factorial(n)
    pi=math.pi
    norm=1./np.sqrt(np.power(2,n)*nfac)/np.power(pi,0.25)

    if( n == 0 ) :
        Hn=np.ones_like(x)
    elif( n==1 ) :
        Hn=2.*x;
    elif( n > 1 ) :
        hm2=np.ones_like(x);
        hm1=2.*x;
        for m in range(2,n+1) :
            hmx=2.*x*hm1 - 2.*np.real(m-1)*hm2
            hm2=hm1
            hm1=hmx
        Hn=hmx
    else :
        print("n is wrong! Exiting code");
        exit()

    wf=norm*np.exp(-np.power(x,2)/2.)*Hn

    return wf
