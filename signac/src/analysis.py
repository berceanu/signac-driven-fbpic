import numpy as np
import matplotlib.pyplot as plt
import pathlib
from openpmd_viewer import OpenPMDTimeSeries


z=ts.get_particle(['z'],species='bunch',iteration=iter) 
x=ts.get_particle(['x'],species='bunch',iteration=iter)  

z = z[0]
x = x[0]

nbz = 200
nbx = 200
zmin = min(z)
zmax = max(z)
xmin = -0.0001
xmax = 0.0001

fig = plt.figure(figsize=(7, 5))
h = plt.hist2d(z,x,bins=(nbz, nbx),range=[[zmin, zmax], [xmin, xmax]])
counts = h[0]
nz = h[1]
nz_r = np.resize(nz, nbz) 
nx = h[2]
nx_r = np.resize(nx, nbx) 
#plt.plot(nx_r,counts[12,:])
 
refval = 0

centroid = []
we = []
centroid_z =[]

#cut_min = 0
#cut_max = 500

for i in range(0,nbz):
    #print(max(counts[i,:]))
    if refval<max(counts[i,:]):
        refval = max(counts[i,:])
        

#print(refval)


for i in range(0,nbz):
    counts[i,:][counts[i,:] < 0.15*refval]=0
    if max(counts[i,:])>0.2*refval and i>20 and i<nbz-20:
        #a = counts[i,:]
        #centroid.append(nx_r(np.argmax(a)))
        centroid.append(np.average(nx_r, weights=counts[i,:]))
        centroid_z.append(nz_r[i])
        we.append(sum(counts[i,:]))

plt.plot(centroid_z,centroid) 
plt.xlabel('z (m)',fontsize=20)
plt.ylabel('x (m)',fontsize=20)

mean_head = np.mean(centroid[len(centroid)-3:len(centroid)])
print(abs(np.mean(centroid)-mean_head))
print(abs(np.average(centroid,weights=we)-mean_head))
#print(centroid[-1])
 
if __name__ == "__main__":
    p = pathlib.Path('/scratch/berceanu/runs/signac-driven-fbpic/workspace/1f39b6d2d358860226da746c9f362d2d')
    h5_path = p / "diags" / "hdf5"

    ts: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=True)
    iterations = time_series.iterations

    iter = iterations[-1]

