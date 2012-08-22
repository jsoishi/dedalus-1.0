import numpy as np
import pylab as P
import h5py
import glob
import os

ddir = 'snap_00004'
field = '/fields/u/1'
cpus = glob.glob(os.path.join(ddir,'data.cpu*'))

files = []
dd = []
for cpu in cpus:
    files.append(h5py.File(cpu,'r'))
    dd.append(files[-1][field])

data = np.concatenate(dd,axis=0)
print data
for i in range(len(cpus)):
    files[i].close()

print data.real.min(), data.real.max()
data2 =np.fft.fftn(data)
P.imshow(data2[:,0,:].real)
P.colorbar()
P.show()
