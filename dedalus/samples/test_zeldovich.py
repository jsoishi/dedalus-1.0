from dedalus.mods import *
from scipy.integrate import quad
import numpy as na
import pylab as pl
import os

def reorder(arr):
    di = len(arr)/2
    tmp = na.array([arr[i-di] for i in xrange(len(arr))])
    return tmp 

shape = (16,16,16)
RHS = CollisionlessCosmology(shape, FourierRepresentation)
data = RHS.create_fields(0.)
H_0 = 7.185e-5 # 70.3 km/s/Mpc in Myr^-1 (2.27826587e-18 seconds^-1) 

L = 2*na.pi # size of box
N_p = 128 # resolution of analytic solution
q = na.array([i for i in xrange(0,N_p)]) * L / N_p
k = 2*na.pi/L # wavenumber of initial velocity perturbation
a_i = 0.002 # initial scale factor
t0 = (2./3.)/H_0 # present age of E-dS universe
t_init = (a_i**(3./2.)) * t0 # time at which a = a_i in this universe

Omega_r = 8.4e-5
Omega_m = 0.276
Omega_l = 0.724

# ... growth factor calculations
H = lambda ap: H_0*na.sqrt(Omega_r/ap**4 + Omega_m/ap**3 + 
                           (1-Omega_r-Omega_l-Omega_m)/ap**2 + Omega_l)
Hint = lambda ap: 1./(ap**3)/(H(ap))**3
D_unnorm = lambda ap: H_0**2 * H(ap) * quad(Hint, 0, ap, epsabs=1.0e-10)[0]
D0 = D_unnorm(a_i)
Ddot_i = (H(a_i)*a_i)*(D_unnorm(a_i + 1e-6) - D0)/(1e-6)/D0

a_cross = 0.1
A = D0/(H_0**2 * H(a_cross) * quad(Hint, 0, a_cross, epsabs=1.0e-10)[0] * k)
ampl = A * a_i * Ddot_i

print "a_cross = ", a_cross
tcross = (a_cross**(3./2.))*t0 # valid for Einsten-de Sitter 

RHS.parameters['Omega_r'] = Omega_r
RHS.parameters['Omega_m'] = Omega_m
RHS.parameters['Omega_l'] = Omega_l
RHS.parameters['H0'] = H_0
zeldovich(data, ampl, A, a_i, a_cross)

Myr = 1 # 3.15e13 seconds
tstop = tcross - t_init
dt = Myr

ti = RK2simple(RHS)
ti.stop_time(tstop)

ddelta = []
uu = []
uk = []

t_snapshots = []
a_snapshots = []
  
an = AnalysisSet(data, ti)
an.add("field_snap", 20)
an.add("en_spec", 20)

i = 0
#an.run()
while ti.ok:
    print "step: ", i, " a = ", RHS.aux_eqns['a'].value
    if i % 80 == 0:
        tmp = na.zeros_like(data['u'][0]['xspace'][0,0,:].real)
        tmp[:] = data['u'][0]['xspace'][0,0,:].real
        
        tmp2 = na.zeros_like(data['delta']['xspace'][0,0,:].real)
        tmp2[:] = data['delta']['xspace'][0,0,:].real

        tmp3 = na.zeros_like(data['u'][0]['kspace'][0,0,:].real)
        tmp3[:] = na.abs(data['u'][0]['kspace'][0,0,:])
        
        uu.append(tmp)
        ddelta.append(tmp2)
        uk.append(tmp3)
        t_snapshots.append(data.time)
        a_snapshots.append(RHS.aux_eqns['a'].value)
    ti.advance(data, dt)
    #an.run()
    i = i + 1

tmp = na.zeros_like(data['u'][0]['xspace'][0,0,:].real)
tmp[:] = data['u'][0]['xspace'][0,0,:].real

tmp2 = na.zeros_like(data['delta']['xspace'][0,0,:].real)
tmp2[:] = data['delta']['xspace'][0,0,:].real

tmp3 = na.zeros_like(data['u'][0]['kspace'][0,0,:].real)
tmp3[:] = na.abs(data['u'][0]['kspace'][0,0,:])

uu.append(tmp)
ddelta.append(tmp2)
uk.append(tmp3)
t_snapshots.append(data.time)
a_snapshots.append(RHS.aux_eqns['a'].value)
print "a_stop = ", RHS.aux_eqns['a'].value

x_grid = na.array([i for i in xrange(shape[0])])*data.length[0]/shape[0]

if not os.path.exists('frames'):
    os.mkdir('frames')

fig = pl.figure()
for i,a in enumerate(a_snapshots):

    # Compare to smooth analytic solution
    outfile = "frames/cmp_a%05f.png" % a

    # ... growth factor
    Da = D_unnorm(a)
    D = Da/D0
    da = 1e-6
    Ddot = (H(a)*a)*(D_unnorm(a + da) - Da)/da/D0

    # ... analytic solution
    x = q + D*A*na.sin(k*q)
    delta = 1./(1.+D*A*k*na.cos(k*q))-1.
    v = a * Ddot * A * na.sin(k*q)
    #phi = 3/2/a * ( (q**2 - x**2)/2 + 
    #                D * A * k * (k*q*na.sin(k*q) + na.cos(k*q) - 1) )
    pl.plot(x, v)
    pl.plot(x_grid, reorder(uu[i]), '.', hold=True)
    pl.title('a = %05f' % a)
    fig.savefig(outfile)
    fig.clf()

    pl.plot(x, delta)
    pl.plot(x_grid, reorder(ddelta[i]), '.', hold=True)
    pl.title('a = %05f' % a)
    fig.savefig("frames/delta_a%05f.png" % a)
    fig.clf()
    
    # Residuals
    #q_grid = na.array(broyden1(lambda y: na.array(y) + 
    #                          D*A*na.sin(k*na.array(y)) - x_grid,x_grid))
    #v_grid = a * Ddot * A * na.sin(k*q_grid)
    #resid = v_grid - reorder(uu[i])
    
    #pl.plot(x_grid, resid, '.')
    #outfile = "frames/res_a%05f.png" % a
    #pl.title('a = %05f' % a)
    #fig.savefig(outfile)
    #fig.clf()
