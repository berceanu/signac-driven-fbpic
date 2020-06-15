import numpy as np
import matplotlib.pyplot as plt 

def dens_func_ext( z, r , filedata):
    # Allocate relative density
    n = np.ones_like(z)
    # Read density data and nomalize
    l = np.loadtxt(filedata, usecols=(0)) 
    ne = np.loadtxt(filedata, usecols=(1))
    l = (l - l[0])*1e-4
    ne = ne/np.amax(ne)
    # Take only some point to keep the profile not too irregular 
    numElems = 56
    idx = np.round(np.linspace(0, len(l) - 1, numElems)).astype(int)
    l = l[idx]
    ne = ne[idx]
    # Interpolate data
    n = np.interp(z,l,ne)
    return(n)

xvals = np.linspace(0, 3000*1e-4, 1000)
ne = dens_func_ext(xvals,0,'density_16.txt')
plt.plot(xvals, ne, 'o')
plt.show()
                                   
    


