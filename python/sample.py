from physics import Hydro
from time_integration import RK2
from init_cont import turb
from analysis import Analysis

shape = (128,128,128)
RHS = Hydro(shape)

ti = RK2(RHS,courant=0.4)
ti.stop_time(10.) # set stoptime
ti.stop_iter(1000) # max iterations
ti.stop_walltime(10.) # stop after 10 hours
an = Analysis(RHS)

#main loop
while ti.ok():
    ti.advance()
    an.chk() # see if it's time for analysis

ti.final_stats()

print "Done"
