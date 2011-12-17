import numpy as na
import pylab as P
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myproc = comm.Get_rank()
x = na.linspace(0,10,100)
xx,yy = na.meshgrid(x,x)
nz_local = 100/nproc
data = yy[myproc*nz_local:(myproc+1)*nz_local,:]
incoming = na.empty((100,nz_local))

size = []
soffset = []
roffset = []
for i in range(nproc):
    size.append(data.size/nproc)
    soffset.append(i*size[i])
    roffset.append(i*size[i])
print size, soffset
send = data
comm.Alltoallv((send.ravel('F'), (size, soffset), MPI.DOUBLE), 
               (incoming, (size,roffset), MPI.DOUBLE))


P.subplot(121)
P.imshow(incoming,vmin=0, vmax=10)
P.title("Proc %i" % myproc)
P.colorbar()
P.subplot(122)
P.imshow(send,vmin=0,vmax=10)
#P.plot(incoming[:,0])
P.colorbar()
P.show()
