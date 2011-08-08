"""Time Integrators. 

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

  This file is part of dedalus.

  dedalus is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import cPickle
import time
import h5py
import numpy as na

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

from dedalus.funcs import insert_ipython, get_mercurial_changeset_id

class TimeStepBase(object):
    def __init__(self, RHS, CFL=0.4, int_factor=None):
        """Base class for dedalus time stepping methods. Provides
        stopping controls, statistics, and (eventually) a snapshotting
        trigger.

        """
        self.RHS = RHS
        self.CFL = CFL
        self.int_factor = int_factor
        self._is_ok = True
        self._nsnap = 0
        self._tlastsnap = 0.

        # parameters
        self.iter = 0
        self.time = 0.
        self._stop_iter = 100000
        self._stop_time = -1.
        self._stop_wall = 3600.*24. # 24 hours
        self._dnsnap  = 100
        self._dtsnap = 1.

        self.__start_time = time.time()

    @property
    def ok(self):
        if self.iter >= self._stop_iter:
            print "Maximum number of iterations reached. Done."
            self._is_ok = False
        if self.time >= self._stop_time:
            print "Time > stop time. Done"
            self._is_ok = False
        if (time.time() - self.__start_time) >= self._stop_wall:
            print "Wall clock time exceeded. Done."
            self._is_ok = False

        return self._is_ok

    def advance(self, data, dt):
        if ((self.iter % self._dnsnap) == 0) or (data.time - self._tlastsnap >= self._dtsnap):
            self.snapshot(data)
        self.do_advance(data,dt)

    def do_advance(self, data, dt):
        raise NotImplementedError("do_advance must be provided by subclass.")

    def snapshot(self, data):
        if comm:
            myproc = comm.Get_rank()
        else:
            myproc = 0

        pathname = "snap_%05i" % (self._nsnap)
        if not os.path.exists(pathname) and myproc == 0:
            os.mkdir(pathname)
        
        comm.Barrier()
        # first, pickle physics data
        obj_file = open(os.path.join(pathname,'dedalus_obj_%04i.cpkl' % myproc),'w')
        cPickle.dump(self.RHS, obj_file)
        cPickle.dump(data, obj_file)
        cPickle.dump(self, obj_file)
        obj_file.close()

        # now save fields
        filename = os.path.join(pathname, "data.cpu%04i" % myproc)
        outfile = h5py.File(filename, mode='w')
        root_grp = outfile.create_group('/fields')
        dset = outfile.create_dataset('time',data=self.time)
        outfile.attrs['hg_version'] = get_mercurial_changeset_id()

        data.snapshot(root_grp)        
        outfile.close()
        self._nsnap += 1
        self._tlastsnap = self.time
        
    def stop_time(self,t):
        self._stop_time = t

    def stop_iter(self, it):
        self._stop_iter = it

    def stop_walltime(self, wt):
        self._stop_wall = wt

    def set_nsnap(self, n):
        self._dnsnap = n

    def set_dtsnap(self, dt):
        self._dtsnap = dt
                       
    def final_stats(self):
        stop_time = time.time()
        total_wtime = stop_time - self.__start_time
        print "total wall time: %10.5e sec" % total_wtime
        print "%10.5e sec/step " %(total_wtime/self.iter)
        print "Simulation complete. Status: awesome"


class RK2simple(TimeStepBase):
    """Basic second-order (midpoint) Runga-Kutta integrator."""
        
    def __init__(self, *arg, **kwargs):        
        TimeStepBase.__init__(self, *arg, **kwargs)
        
        # Create StateData for storing stage
        self.tmp_fields = self.RHS.create_fields(0.)
        
        # Create internal reference to derivative StateData from RHS
        self.field_dt = None

    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
          
          Peyret 143 for high order
        """
        
        # First step
        self.field_dt = self.RHS.RHS(data)
        linear_step(data, self.field_dt, dt / 2., self.tmp_fields)

        for a in self.RHS.aux_eqns.values():
            a_old = a.value # OK if we only have one aux eqn...
            # need to update actual value so RHS can use it
            a.value = a.value + dt / 2. * a.RHS(a.value)

        # Second step
        self.field_dt = self.RHS.RHS(self.tmp_fields)
        linear_step(data, self.field_dt, dt, data)

        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a.RHS(a.value)
                     
        # Update integrator stats
        self.time += dt
        self.iter += 1
        for k,f in data.fields.iteritems():
            self.tmp_fields[k].zero_all() 
            self.field_dt[k].zero_all()

class RK2simplevisc(RK2simple):
    """Midpoint RK2 with integrating factors."""
    
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """

        # First step
        self.field_dt = self.RHS.RHS(data)
        integrating_factor_step(data, self.field_dt, dt / 2., self.tmp_fields)
              
        for a in self.RHS.aux_eqns.values():
            # OK if we only have one aux eqn...
            # need to update actual value so RHS can use it
            a_old = a.value 
            a.value = a.value + dt / 2. * a.RHS(a.value)

        # Second step
        self.field_dt = self.RHS.RHS(self.tmp_fields)
        integrating_factor_step(data, self.field_dt, dt, data)

        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a.RHS(a.value)

        # Update integrator stats
        self.time += dt
        self.iter += 1
        
class RK4simplevisc(RK2simple):
    """Standard RK4 with integrating factors."""
    
    def __init__(self, *arg, **kwargs):        
        TimeStepBase.__init__(self, *arg, **kwargs)
        
        # Create StateData for storing stages
        self.tmp_stage = self.RHS.create_fields(0.)
        
        # Create StateData for building final step
        self.tmp_final = self.RHS.create_fields(0.)
        
        # Create internal reference to derivative StateData from RHS
        self.field_dt = None

    
    def do_advance(self, data, dt):
        """
        k1 = RHS(t_n, y_n)
        k2 = RHS(t_n + h / 2, y_n + dt / 2 * k1)
        k3 = RHS(t_n + h / 2, y_n + dt / 2 * k2)
        k4 = RHS(t_n + h, y_n + dt * k3)
          
        y_n+1 = y_n + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6 + O(h ** 5)
        t_n+1 = t_n + h
        """
        
        # First stage
        self.field_dt = self.RHS.RHS(data)
        
        for k,f in data.fields.iteritems():
            for i in xrange(f.ncomp):
                self.tmp_final[k][i]['kspace'] = self.field_dt[k][i]['kspace'] / 6.
                
        for a in self.RHS.aux_eqns.values():
            # OK if we only have one aux eqn...
            # need to update actual value so RHS can use it
            a_old = a.value 
            a_final_dt = a.RHS(a.value) / 6.
            a.value = a_old + dt / 2. * a.RHS(a.value)
                
        # Second stage
        integrating_factor_step(data, self.field_dt, dt / 2., self.tmp_stage)
        self.field_dt = self.RHS.RHS(self.tmp_stage)
        
        for k,f in data.fields.iteritems():
            for i in xrange(f.ncomp):
                self.tmp_final[k][i]['kspace'] += self.field_dt[k][i]['kspace'] / 3.
                
        for a in self.RHS.aux_eqns.values():
            a_final_dt += a.RHS(a.value) / 3.
            a.value = a_old + dt / 2. * a.RHS(a.value)
                
        # Third stage
        integrating_factor_step(data, self.field_dt, dt / 2., self.tmp_stage)
        self.field_dt = self.RHS.RHS(self.tmp_stage)
                       
        for k,f in data.fields.iteritems():
            for i in xrange(f.ncomp):
                self.tmp_final[k][i]['kspace'] += self.field_dt[k][i]['kspace'] / 3.
                
        for a in self.RHS.aux_eqns.values():
            a_final_dt += a.RHS(a.value) / 3.
            a.value = a_old + dt * a.RHS(a.value)
                
        # Fourth stage
        integrating_factor_step(data, self.field_dt, dt, self.tmp_stage)
        self.field_dt = self.RHS.RHS(self.tmp_stage)
                               
        for k,f in data.fields.iteritems():
            for i in xrange(f.ncomp):
                self.tmp_final[k][i]['kspace'] += self.field_dt[k][i]['kspace'] / 6.
                
        for a in self.RHS.aux_eqns.values():
            a_final_dt += a.RHS(a.value) / 6.
        
        # Final step
        integrating_factor_step(data, self.tmp_final, dt, data)

        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a_final_dt

        # Update integrator stats
        self.time += dt
        self.iter += 1

class CrankNicholsonVisc(TimeStepBase):
    """
    Crank-Nicholson timestepping, copied from MIT script,
    mit18336_spectral_ns2d.m

    """
    
    def do_advance(self, data, dt):
    
        deriv = self.RHS.RHS(data)
        for k,f in deriv.fields.iteritems():
            top = 1. / dt - 0.5 * f.integrating_factor
            bottom = 1. / dt + 0.5 * f.integrating_factor
            for i in xrange(f.ncomp):
                data[k][i]['kspace'] = (top / bottom * data[k][i]['kspace'] + 
                                        1. / bottom * deriv[k][i]['kspace'])

        # Update data and integrator stats
        data.set_time(data.time + dt)
        self.time += dt
        self.iter += 1
            
def linear_step(start, deriv, dt, output):
    """
    Take a linear step in all fields.
    
    Inputs:
        start       StateData object at initial time
        deriv       StateData object containing derivatives
        dt          Timestep
        output      StateData object to take the output
        
    """
    
    output.set_time(start.time + dt)
    
    for k,f in start.fields.iteritems():
        for i in xrange(f.ncomp):
            output[k][i]['kspace'] = start[k][i]['kspace'] + dt * deriv[k][i]['kspace']
            output[k][i].dealias()
            
                
def integrating_factor_step(start, deriv, dt, output):
    """
    Take a step using integrating factors in all fields.
    
    Inputs:
        start       StateData object at initial time
        deriv       StateData object containing derivatives
        dt          Timestep
        output      StateData object to take the output
        
    """
    
    output.set_time(start.time + dt)
    
    for k,f in start.fields.iteritems():
        # Exponentiate the integrating factor
        IF = deriv[k].integrating_factor
        
        # Turn zeros into small numbers: to first order, reduces to linear step
        if IF == None:
            IF = 1e-10
        else:
            IF[IF == 0] = 1e-10
        EIF = na.exp(IF * dt)
            
        for i in xrange(f.ncomp):
            output[k][i]['kspace'] = (start[k][i]['kspace'] + deriv[k][i]['kspace'] / IF * (EIF - 1.)) / EIF
            output[k][i].dealias()

    
    
