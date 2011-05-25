from physics import Hydro
from fourier_data import FourierData
from time_stepper import RK2simple
from init_cond import taylor_green
#from analysis import Analysis

shape = (100,100) #(128,128,128)
RHS = Hydro(shape, FourierData)
data = RHS.create_fields(0.)
taylor_green(data['ux'],data['uy'])

ti = RK2simple(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(10) # max iterations
ti.stop_walltime(10.) # stop after 10 hours
#an = Analysis(RHS)

#main loop
dt = 1e-6
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    #an.chk() # see if it's time for analysis

ti.final_stats()

print "Done"
