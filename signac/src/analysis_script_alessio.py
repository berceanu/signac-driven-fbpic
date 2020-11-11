import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

# Replace the string below, to point to your data
ts = OpenPMDTimeSeries('./7e13/diags/hdf5/')

z=ts.get_particle(['z'],iteration=21000) 
x=ts.get_particle(['x'],iteration=21000)  

z = z[0]
x = x[0]

nbz = 200
nbx = 200
zmin = min(z)#max(z)-0.003*3
zmax = max(z)
#xmin = min(x)
#xmax = max(x)
xmin = -0.001
xmax = 0.001

fig = plt.figure(figsize=(7, 5))
h = plt.hist2d(z,x,bins=(nbz, nbx),range=[[zmin, zmax], [xmin, xmax]])
counts = h[0]
nz = h[1]
nz_r = np.resize(nz, nbz) 
nx = h[2]
nx_r = np.resize(nx, nbx) 
#plt.plot(nx_r,counts[12,:])
#plt.show()
 
refval = 0

centroid = []
we = []
centroid_z =[]

cut_min = 0
cut_max = 500

for i in range(0,nbz):
    #print(max(counts[i,:]))
    if refval<max(counts[i,:]):
        refval = max(counts[i,:])
        

#print(refval)


for i in range(0,nbz):
    counts[i,:][counts[i,:] < 8]=0
    if max(counts[i,:])>0.3*refval and i>20 and i<180:
        #a = counts[i,:]
        #centroid.append(nx_r(np.argmax(a)))
        centroid.append(np.average(nx_r, weights=counts[i,:]))
        centroid_z.append(nz_r[i])
        we.append(sum(counts[i,:]))

plt.plot(centroid_z,centroid) 
plt.xlabel('z (m)',fontsize=20)
plt.ylabel('x (m)',fontsize=20)
plt.show()

mean_head = np.mean(centroid[len(centroid)-5:len(centroid)])
print(abs(np.mean(centroid)-mean_head))
print(abs(np.average(centroid,weights=we)-mean_head))
#print(centroid[-1])

 
