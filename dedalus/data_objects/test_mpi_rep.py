from fourier_data import ParallelFourierRepresentation
from mpi4py import MPI
from numpy import pi
comm = MPI.COMM_WORLD

nproc = comm.Get_size()
print nproc
fr = ParallelFourierRepresentation(None, [16/nproc,16,16], [2*pi/nproc,2*pi,2*pi])

outfile = open("data_%s" % comm.Get_rank(),'w')
outfile.write(str(fr.k['z']))
outfile.close()
