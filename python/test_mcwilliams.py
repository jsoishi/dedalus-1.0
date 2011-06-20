from physics import Hydro
from fourier_data import FourierData
from time_step import RK2simple,RK2simplevisc, RK2simplehypervisc4
from init_cond import turb, mcwilliams_spec
from analysis import AnalysisSet

shape = (450,450)
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 3.5e-7 # 100x mcwilliams
data = RHS.create_fields(0.)

turb(data['ux'],data['uy'],mcwilliams_spec,k0=23.)
ti = RK2simplehypervisc4(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(100) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours

an = AnalysisSet(data, ti)
an.add("print_energy", 1)
an.add("field_snap", 10)
an.add("en_spec",5)
#main loop
dt = 2.5e-3
#snapshot(data,0)
an.run()
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()