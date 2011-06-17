import pylab as P
from mpl_toolkits.axes_grid1 import AxesGrid
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
def field_snap(data, it):
    """take a snapshot of all fields defined. currently only works in
    2D; it will need a slice index for 3D.

    """
    fields = data.fields.keys()
    fields.sort()
    nvars = len(fields)
    nrow = nvars / 3
    if nrow == 0:
        ncol = nvars % 3
        nrow = 1
    else:
        ncol = 3
    fig = P.figure(1,figsize=(24.*ncol/3.,24.*nrow/3.))
    grid = AxesGrid(fig,111,
                    nrows_ncols = (nrow, ncol),
                    axes_pad=0.1,
                    cbar_pad=0.,
                    share_all=True,
                    label_mode="1",
                    cbar_location="top",
                    cbar_mode="each")
    for i,f in enumerate(fields):
        im = grid[i].imshow(data[f]['xspace'].real, cmap='bone')
        grid[i].text(0.05,0.95,f, transform=grid[i].transAxes, size=24,color='white')
        grid.cbar_axes[i].colorbar(im)

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/snap_%04i.png" % it
    fig.savefig(outfile)
    fig.clf()

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

