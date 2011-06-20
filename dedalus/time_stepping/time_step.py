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
from yt.funcs import insert_ipython
class TimeStepBase(object):
    def __init__(self, RHS, CFL=0.4, int_factor=None):
        """a very simple Runga-Kutta 2 integrator

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
        self._dtsnap = 0.

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
        if (self.iter % self._dnsnap) or (data.time - self._tlastsnap >= self._dtsnap):
            #data.snapshot(self._nsnap)
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

    def final_stats(self):
        stop_time = time.time()
        print "total wall time: %10.5e sec" % (stop_time - self.__start_time)
        print "Simulation complete. Status: awesome"


class RK2simple(TimeStepBase):
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        tmp_fields = self.RHS.create_fields(data.time)
        field_dt = self.RHS.create_fields(data.time)

        # first step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(data)[f]['kspace']
            tmp_fields[f] = data[f]['kspace'] + dt/2. * field_dt[f]['kspace']
        tmp_fields.time = data.time + dt/2.

        # second step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(tmp_fields)[f]['kspace']
            data[f] = data[f]['kspace'] + dt * field_dt[f]['kspace']
        data.time += dt
        self.time += dt
        self.iter += 1

class RK2simplevisc(TimeStepBase):
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        tmp_fields = self.RHS.create_fields(data.time)
        field_dt = self.RHS.create_fields(data.time)

        k2 = na.zeros(data['ux'].data.shape)
        for k in data['ux'].k.values():
            k2 += k**2

        # first step
        viscosity = na.exp(-k2*dt/2.*self.RHS.parameters['nu'])
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(data)[f]['kspace']
            tmp_fields[f] = (data[f]['kspace'] + dt/2. * field_dt[f]['kspace'])*viscosity
            
        tmp_fields.time = data.time + dt/2.

        # second step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(tmp_fields)[f]['kspace']
            data[f] = (data[f]['kspace'] * viscosity + dt * field_dt[f]['kspace'])*viscosity
        data.time += dt
        self.time += dt
        self.iter += 1

class RK2simplehypervisc4(TimeStepBase):
    def do_advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        tmp_fields = self.RHS.create_fields(data.time)
        field_dt = self.RHS.create_fields(data.time)

        k4 = na.zeros(data['ux'].data.shape)
        for k in data['ux'].k.values():
            k4 += k**4

        # first step
        viscosity = na.exp(-k4*dt/2.*self.RHS.parameters['nu'])
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(data)[f]['kspace']
            tmp_fields[f] = (data[f]['kspace'] + dt/2. * field_dt[f]['kspace'])*viscosity
            
        tmp_fields.time = data.time + dt/2.

        # second step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(tmp_fields)[f]['kspace']
            data[f] = (data[f]['kspace'] * viscosity + dt * field_dt[f]['kspace'])*viscosity
        data.time += dt
        self.time += dt
        self.iter += 1