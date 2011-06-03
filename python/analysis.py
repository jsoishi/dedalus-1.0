import pylab as P
import numpy as na
import os
from functools import wraps

class AnalysisSet(object):
    known_analysis = {}
    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, name, cadence):
        self.tasks.append((self.known_analysis[name], cadence))

    def run(self):
        for f, c in self.tasks:
            if self.ti.iter % c != 0: continue
            f(self.data, self.ti.iter)

    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func


@AnalysisSet.register_task
def snapshot(data, it):
    P.subplot(121)
    P.title('t = %5.2f' % data.time)
    P.imshow(data['ux']['xspace'].real)#, vmin=-1, vmax=1)
    P.colorbar()
    P.subplot(122)
    P.imshow(data['uy']['xspace'].real)#, vmin=-1, vmax=1)
    P.colorbar()
    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/snap_%04i.png" % it
    P.savefig(outfile)
    P.clf()

@AnalysisSet.register_task
def print_energy(data, it):
    """compute energy in real space

    """

    energy = na.zeros(data['ux']['xspace'].shape)
    for f in data.fields:
        energy += (data[f]['xspace']*data[f]['xspace'].conj()).real

    print "energy: %10.5e" % (energy.sum()/(8.*na.pi**2))

@AnalysisSet.register_task
def en_spec(data, it):
    kk = na.zeros(data['ux'].data.shape)
    for k in data['ux'].k.values():
        kk += k**2
    kk = na.sqrt(kk)
    power = na.zeros(data['ux'].data.shape)
    for f in data.fields:
        power += (data[f]['kspace']*data[f]['kspace'].conj()).real
    
    nbins = (data['ux'].k['x'].size)/2 
    k = na.arange(nbins)
    spec = na.zeros(nbins)
    for i in range(nbins):
        #spec[i] = (4*na.pi*i**2*power[(kk >= (i-1/2.)) & (kk <= (i+1/2.))]).sum()
        spec[i] = (power[(kk >= (i-1/2.)) & (kk <= (i+1/2.))]).sum()

    P.loglog(k,spec)
    from init_cond import mcwilliams_spec
    P.loglog(k,mcwilliams_spec(k,30.,0.5))
    P.ylim(1e-20,1e-1)
    P.xlabel(r"$k$")
    P.ylabel(r"$E(k)$")

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/enspec_%04i.png" % it
    P.savefig(outfile)
    P.clf()

