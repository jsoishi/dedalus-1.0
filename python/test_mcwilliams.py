from physics import Hydro
from fourier_data import FourierData
from time_stepper import RK2simple,RK2simplevisc
from init_cond import turb, mcwilliams_spec
from analysis import AnalysisSet

shape = (450,450)
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 0.1
data = RHS.create_fields(0.)

turb(data['ux'],data['uy'],mcwilliams_spec,k0=30, ampl=0.5)
ti = RK2simplevisc(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(1) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours
#an = Analysis(RHS)
from yt.funcs import insert_ipython
insert_ipython()

an = AnalysisSet(data, ti)
an.add("print_energy", 1)
an.add("snapshot", 100)
an.add("en_spec",1)
#main loop
dt = 1e-3
#snapshot(data,0)
an.run()
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    an.run()
    #print "ux (min, max) = (%10.5e, %10.5e, %10.5e, %10.5e)" % ((data['ux']['xspace'].real).min(),(data['ux']['xspace'].imag).min(), (data['ux']['xspace'].real).max(), (data['ux']['xspace'].imag).max())
    #print "uy (min, max) = (%10.5e, %10.5e, %10.5e, %10.5e)" % ((data['uy']['xspace'].real).min(),(data['uy']['xspace'].imag).min(), (data['uy']['xspace'].real).max(), (data['uy']['xspace'].imag).max())

    #an.chk() # see if it's time for analysis

ti.final_stats()
