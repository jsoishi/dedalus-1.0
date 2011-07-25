from dedalus.mods import *
import numpy as na
import sys
import pylab as pl

if len(sys.argv) != 3:
    print "usage: ", sys.argv[0], " <ic data file> <normalization data file>"
    print """sample ic and normalization data are included in files
ic_input.txt (generated by linger++, linger-mode output with parameters in ic_param.inp)
norm_input.txt (generated by linger++, transfer-mode output)
"""
    sys.exit()

icfname = sys.argv[1]
normfname = sys.argv[2]

shape = (32,32,32)
kny = 4.
L = 32*na.pi/kny
RHS = CollisionlessCosmology(shape, FourierRepresentation, length=L)
data = RHS.create_fields(0.)
H0 = 7.185e-5 # 70.3 km/s/Mpc in Myr
a_i = 0.002 # initial scale factor
t0 = (2./3.)/H0 # present age of E-dS universe
t_ini = (a_i**(3./2.)) * t0 # time at which a = a_i

def pow_spec(data, it, Dplus):
    delta = data['delta']

    power = na.abs(delta['kspace'])**2

    kmag = na.sqrt(delta.k2())
    k = delta.k['x'].flatten()
    k = na.abs(k[0:(k.size / 2 + 1)])
    kbottom = k - k[1] / 2.
    ktop = k + k[1] / 2.
    spec = na.zeros_like(k)

    for i in xrange(k.size):
        spec[i] = (power[(kmag >= kbottom[i]) & (kmag < ktop[i])]).sum()/(Dplus*Dplus)
        outfile = "frames/powspec_%d.png" % it
    fig = pl.figure()
    pl.semilogy(k[1:], spec[1:], 'o-')
    pl.xlabel("$k$")
    pl.ylabel("$\mid \delta_k \mid^2$")
    fig.savefig(outfile)

RHS.parameters['Omega_r'] = 0#8.4e-5
RHS.parameters['Omega_m'] = 1#0.276
RHS.parameters['Omega_l'] = 0#0.724
RHS.parameters['H0'] = H0
cosmology(data, icfname, normfname)

dt = 1./64 # time in Myr
ti = RK2simple(RHS)
ti.stop_time(100.*dt)

an = AnalysisSet(data, ti)

an.add("field_snap", 20)
#an.add("en_spec", 20)
i=0
an.run()
while ti.ok:
    Dplus = ((data.time + t_ini)/t_ini) ** (2./3.)
    print 'step: ', i
    ti.advance(data, dt)
    an.run()
    if i % 20 == 0:
        pow_spec(data, i, Dplus)
    i = i + 1
