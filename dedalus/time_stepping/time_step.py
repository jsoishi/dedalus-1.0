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
import inspect
import os
import cPickle
import time
import h5py
import numpy as na

from dedalus.utils.parallelism import com_sys
from dedalus.utils.logger import mylog
from dedalus.funcs import insert_ipython, get_mercurial_changeset_id
from dedalus.utils.timer import timer

try:
    from dedalus.__hg_version__ import hg_version
except ImportError:
    hg_version = get_mercurial_changeset_id()
except:
    mylog.warning("could not find hg_version.")
    hg_version = "unknown"

class TimeStepBase(object):
    timer = timer
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

        self._start_time = time.time()

        # Import Cython step functions
        if self.RHS.ndim == 2:
            from forward_step_cy_2d import linear_step, intfac_step
        else:
            from forward_step_cy_3d import linear_step, intfac_step
        self.linear_step = linear_step
        self.intfac_step = intfac_step

    @property
    def ok(self):
        if self.iter >= self._stop_iter:
            if com_sys.myproc == 0:
                mylog.info("Maximum number of iterations reached. Done.")
            self._is_ok = False
        elif self.time >= self._stop_time:
            if com_sys.myproc == 0:
                mylog.info("Time > stop time. Done")
            self._is_ok = False
        elif (time.time() - self._start_time) >= self._stop_wall:
            if com_sys.myproc == 0:
                mylog.info("Wall clock time exceeded. Done.")
            self._is_ok = False
        else:
            self._is_ok = True

        return self._is_ok
    @timer
    def advance(self, data, dt):
        if ((self.iter % self._dnsnap) == 0) or (data.time - self._tlastsnap >= self._dtsnap):
            self.snapshot(data)
        self.do_advance(data,dt)
        mylog.info("step %i" % self.iter)
    def do_advance(self, data, dt):
        raise NotImplementedError("do_advance must be provided by subclass.")

    @timer
    def snapshot(self, data):
        myproc = com_sys.myproc

        pathname = "snap_%05i" % (self._nsnap)
        if not os.path.exists(pathname) and myproc == 0:
            os.mkdir(pathname)
        if com_sys.comm:
            com_sys.comm.Barrier()

        # first, save forcing functions, if any
        string = ""
        for k,forcer in self.RHS.forcing_functions.iteritems():
            if forcer:
                string += inspect.getsource(forcer) + "\n"
                self.RHS._forcing_function_names[k] = forcer.__name__

        sidecar = open(os.path.join(pathname,'forcing_functions.py'),'w')
        sidecar.writelines(string)
        sidecar.close()


        # next, pickle physics data
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
        outfile.attrs['hg_version'] = hg_version

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
        self.timer.print_stats()
        if com_sys.myproc == 0:
            total_wtime = self.timer.timers['advance']
            print "total advance wall time: %10.5e sec" % total_wtime
            print "%10.5e sec/step " %(total_wtime/self.iter)
            print
            print "Simulation complete. Status: awesome"

    def finalize(self, data):
        """helper function to call at the end of a run to clean up,
        write final output, and print stats.

        """
        self.snapshot(data)
        self.final_stats()

    @timer
    def forward_step(self, start, deriv, output, dt):
        """
        Take a step in all fields using forward-Euler for components without
        integrating factors.  Compiled in Cython for speed.

        Parameters
        ----------
        start : StateData object
            Data at initial time
        deriv : StateData object
            Time derivatives
        output : StateData object
            Container for output
        dt : float
            Timestep

        """

        # Loop over field components in output
        for fname, field in output:
            for cindex, comp in field:
                if deriv[fname][cindex].integrating_factor is None:
                    self.linear_step(start[fname][cindex]['kspace'],
                                     deriv[fname][cindex]['kspace'],
                                     comp['kspace'],
                                     dt)
                else:
                    self.intfac_step(start[fname][cindex]['kspace'],
                                     deriv[fname][cindex]['kspace'],
                                     comp['kspace'],
                                     deriv[fname][cindex].integrating_factor,
                                     dt)

        # Update time
        output.set_time(start.time + dt)

class RK2mid(TimeStepBase):
    """
    Second-order explicit midpoint Runge-Kutta method.

    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h / 2, y_n + k1 / 2)

    y_(n+1) = y_n + k2 + O(h ** 3)

      0  |  0    0
     1/2 | 1/2   0
    ----------------
         |  0    1

    """

    def __init__(self, *arg, **kwargs):
        TimeStepBase.__init__(self, *arg, **kwargs)

        # Create StateData for constructing increments
        self.deriv = self.RHS.create_fields(0.)
        self.temp_data = self.RHS.create_fields(0.)

    def do_advance(self, data, dt):

        # Construct k1
        self.RHS.RHS(data, self.deriv)

        # Store initial integrating factors for the complete step
        for fname, field in self.temp_data:
            for cindex, comp in field:
                if comp._static_k or (self.deriv[fname][cindex].integrating_factor is None):
                    comp.integrating_factor = self.deriv[fname][cindex].integrating_factor
                else:
                    comp.integrating_factor = self.deriv[fname][cindex].integrating_factor.copy()

        # Construct k2
        self.forward_step(data, self.deriv, self.temp_data, dt / 2.)
        for a in self.RHS.aux_eqns.values():
            a_old = a.value
            a.value = a_old + dt / 2. * a.RHS(a.value)
        self.RHS.RHS(self.temp_data, self.deriv)

        # Retrieve initial integrating factors for the complete step
        for fname, field in self.temp_data:
            for cindex, comp in field:
                self.deriv[fname][cindex].integrating_factor = comp.integrating_factor

        # Complete step
        self.forward_step(data, self.deriv, data, dt)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a.RHS(a.value)

        # Update integrator stats
        self.time += dt
        self.iter += 1


class RK2trap(TimeStepBase):
    """
    Second-order explicit trapezoidal Runge-Kutta method, using exponential
    time differencing to handle integrating factors.

    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h, y_n + k1)

    y_(n+1) = y_n + (k1 + k2) / 2 + O(h ** 3)

      0  |  0    0
      1  |  1    0
    ----------------
         | 1/2  1/2

    """

    def __init__(self, *arg, **kwargs):
        pass


class RK4(TimeStepBase):
    """
    Fourth-order explicit classical Runge-Kutta method.

    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h / 2, y_n + k1 / 2)
    k3 = h * f(t_n + h / 2, y_n + k2 / 2)
    k4 = h * f(t_n + h, y_n + k3)

    y_(n+1) = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + O(h ** 5)

      0  |  0    0    0    0
     1/2 | 1/2   0    0    0
     1/2 |  0   1/2   0    0
      1  |  0    0    1    0
    --------------------------
         | 1/6  1/3  1/3  1/6

    """

    def __init__(self, *arg, **kwargs):
        TimeStepBase.__init__(self, *arg, **kwargs)

        # Create StateData for constructing increments
        self.temp_data = self.RHS.create_fields(0.)

        # Create StateData for combining increments
        self.total_deriv = self.RHS.create_fields(0.)

    def do_advance(self, data, dt):

        # Construct k1
        for a in self.RHS.aux_eqns.values():
            a_old = a.value
            a_final_dt = a.RHS(a.value) / 6.
        deriv = self.RHS.RHS(data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] = deriv[fname][cindex]['kspace'] / 6.

                # Retrieve initial integrating factors for the complete step
                if comp._static_k or (deriv[fname][cindex].integrating_factor is None):
                    comp.integrating_factor = deriv[fname][cindex].integrating_factor
                else:
                    comp.integrating_factor = deriv[fname][cindex].integrating_factor.copy()

        # Construct k2
        self.forward_step(data, deriv, self.temp_data, dt / 2.)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt / 2. * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 3.

        deriv = self.RHS.RHS(self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += deriv[fname][cindex]['kspace'] / 3.

        # Construct k3
        self.forward_step(data, deriv, self.temp_data, dt / 2.)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt / 2. * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 3.

        deriv = self.RHS.RHS(self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += deriv[fname][cindex]['kspace'] / 3.

        # Construct k4
        self.forward_step(data, deriv, self.temp_data, dt)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 6.

        deriv = self.RHS.RHS(self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += deriv[fname][cindex]['kspace'] / 6.

        # Complete step
        self.forward_step(data, self.total_deriv, data, dt)
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

