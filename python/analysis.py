import numpy as na
import os
def snapshot(data, it):
    import pylab as P

    P.subplot(121)
    P.title('t = %5.2f' % data.time)
    P.imshow(data['ux']['xspace'].real, vmin=-1, vmax=1)
    P.colorbar()
    P.subplot(122)
    P.imshow(data['uy']['xspace'].real, vmin=-1, vmax=1)
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
