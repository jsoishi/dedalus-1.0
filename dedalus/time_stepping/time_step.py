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

import time
import numpy as na
from dedalus.funcs import insert_ipython

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
        self._dnsnap  = 1
        self._dtsnap = 10.

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
            data.snapshot(self._nsnap)
            self._nsnap += 1

        self.do_advance(data,dt)

    def do_advance(self, data, dt):
        raise NotImplementedError("do_advance must be provided by subclass.")

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
    def __init__(self, *arg, **kwargs):
        """a very simple Runga-Kutta 2 integrator

        """
        TimeStepBase.__init__(self, *arg, **kwargs)
        self.tmp_fields = self.RHS.create_fields(0.)
        self.field_dt = self.RHS.create_fields(0.)

    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        self.tmp_fields.time = data.time
        self.field_dt.time = data.time
        # first step
        self.field_dt = self.RHS.RHS(data)
        for f in self.RHS.fields:
            self.tmp_fields[f] = data[f]['kspace'] + dt/2. * self.field_dt[f]['kspace']
            self.tmp_fields[f]._curr_space ='kspace'
        for a in self.RHS.aux_eqns.values():
            a_tmp = a.value + dt/2. * a.RHS(a.value)
        self.tmp_fields.time = data.time + dt/2.

        # second step
        self.field_dt = self.RHS.RHS(self.tmp_fields)
        
        for f in self.RHS.fields:
            data[f] = data[f]['kspace'] + dt * self.field_dt[f]['kspace']
            self.tmp_fields[f]._curr_space ='kspace'
        for a in self.RHS.aux_eqns.values():
            a.value = a.value + dt * a.RHS(a_tmp)

        data.time += dt
        self.time += dt
        self.iter += 1
        self.tmp_fields.zero_all() 
        self.field_dt.zero_all()

class RK2simplevisc(RK2simple):
    """Runga-Kutta 2 with integrating factor for viscosity. 

    """
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        self.tmp_fields.time = data.time
        self.field_dt.time = data.time

        k2 = data['ux'].k2()
        # first step
        viscosity = na.exp(-k2*dt/2.*self.RHS.parameters['nu'])
        self.field_dt = self.RHS.RHS(data)
        for f in self.RHS.fields:
            self.tmp_fields[f] = (data[f]['kspace'] + dt/2. * self.field_dt[f]['kspace'])*viscosity
            self.tmp_fields[f]._curr_space ='kspace'
        self.tmp_fields.time = data.time + dt/2.

        # second step
        self.field_dt = self.RHS.RHS(self.tmp_fields)
        for f in self.RHS.fields:
            data[f] = (data[f]['kspace'] * viscosity + dt * self.field_dt[f]['kspace'])*viscosity

        data.time += dt
        self.time += dt
        self.iter += 1
        self.tmp_fields.zero_all() 
        self.field_dt.zero_all()

class RK2simplehypervisc4(RK2simple):
    """Runga-Kutta 2 with integrating factor for 4th order
    hyperviscosity (i.e., \nu_4 \nabla^4 )

    """
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        self.tmp_fields.time = data.time
        self.field_dt.time = data.time

        k4 = na.zeros(data['ux'].data.shape)
        for k in data['ux'].k.values():
            k4 += k**4

        # first step
        viscosity = na.exp(-k4*dt/2.*self.RHS.parameters['nu'])
        self.field_dt = self.RHS.RHS(data)
        for f in self.RHS.fields:
            self.tmp_fields[f] = (data[f]['kspace'] + dt/2. * self.field_dt[f]['kspace'])*viscosity
            self.tmp_fields[f]._curr_space ='kspace'            
        self.tmp_fields.time = data.time + dt/2.
        # second step
        self.field_dt = self.RHS.RHS(self.tmp_fields)
        for f in self.RHS.fields:
            data[f] = (data[f]['kspace'] * viscosity + dt * self.field_dt[f]['kspace'])*viscosity
            self.tmp_fields[f]._curr_space ='kspace'
        data.time += dt
        self.time += dt
        self.iter += 1
        #self.tmp_fields.zero_all() 
        #self.field_dt.zero_all()

class CrankNicholsonVisc(TimeStepBase):
    """Crank-Nicholson timestepping, copied from MIT script,
    mit18336_spectral_ns2d.m

    """
    def do_advance(self, data, dt):
        k2 = data['ux'].k2()
        top = (1./dt - 0.5*self.RHS.parameters['nu']*k2)
        bottom  = (1./dt + 0.5*self.RHS.parameters['nu']*k2)
        deriv = self.RHS.RHS(data)
        for f in deriv.fields:
            data[f] = top/bottom * data[f]['kspace'] + 1./bottom * deriv[f]['kspace']

        data.time += dt
        self.time += dt
        self.iter += 1
