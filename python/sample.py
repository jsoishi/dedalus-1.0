from physics import Hydro
from fourier_data import FourierData
from time_step import RK2simple,RK2simplevisc
from init_cond import taylor_green

shape = (100,100) 
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 0.1
data = RHS.create_fields(0.)

taylor_green(data['ux'],data['uy'])
ti = RK2simplevisc(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(100) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours

#main loop
dt = 1e-3
while ti.ok:
    #print "step: %i" % ti.iter
    ti.advance(data, dt)
ti.final_stats()
