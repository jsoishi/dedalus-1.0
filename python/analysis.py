import pylab as P
import numpy as na
import os

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

def print_energy(data):
    energy = na.zeros(data['ux']['xspace'].shape)
    for f in data.fields:
        energy += (data[f]['xspace']*data[f]['xspace'].conj()).real

    print "energy: %10.5e" % energy.mean()

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
        spec[i] = (power[(kk >= (i-1/2.)) & (kk <= (i+1/2.))]).sum()

    P.loglog(k,spec)
    P.xlabel(r"$k$")
    P.ylabel(r"$E(k)$")

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/enspec_%04i.png" % it
    P.savefig(outfile)
    P.clf()

