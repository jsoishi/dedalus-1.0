from physics import Hydro
from fourier_data import FourierData
from time_stepper import RK2simple,RK2simplevisc
from init_cond import taylor_green
from analysis import snapshot, print_energy

shape = (100,100) #(128,128,128)
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 0.1
data = RHS.create_fields(0.)

taylor_green(data['ux'],data['uy'])
ti = RK2simplevisc(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
#ti.stop_iter(10) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours
#an = Analysis(RHS)

#main loop
dt = 1e-3
snapshot(data,0)
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    if ti.iter % 100 == 0:
        snapshot(data, ti.iter)
    print_energy(data)
    #print "ux (min, max) = (%10.5e, %10.5e, %10.5e, %10.5e)" % ((data['ux']['xspace'].real).min(),(data['ux']['xspace'].imag).min(), (data['ux']['xspace'].real).max(), (data['ux']['xspace'].imag).max())
    #print "uy (min, max) = (%10.5e, %10.5e, %10.5e, %10.5e)" % ((data['uy']['xspace'].real).min(),(data['uy']['xspace'].imag).min(), (data['uy']['xspace'].real).max(), (data['uy']['xspace'].imag).max())

    #an.chk() # see if it's time for analysis

ti.final_stats()
