"""
Time Integrators.

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
import numpy as np

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

    def __init__(self, RHS, CFL=0.1, int_factor=None):
        """
        Base class for dedalus time stepping methods. Provides
        stopping controls, statistics, and (eventually) a snapshotting
        trigger.

        """

        # Store inputs
        self.RHS = RHS
        self.CFL = CFL
        self.int_factor = int_factor

        # Default parameters
        self.sim_stop_time = 1000
        self.wall_stop_time = 60. * 60.
        self.stop_iteration = 1000
        self.save_cadence = 100
        self.max_save_period = 100.

        # Instantiation
        self.time = 0
        self.iteration = 0
        self._nsnap = 0
        self._tlastsnap = 0.
        self.dt_old = np.finfo('d').max / 10.
        self._start_time = time.time()

    @property
    def ok(self):

        if self.iteration >= self.stop_iteration:
            ok_flag = False
            if com_sys.myproc == 0:
                mylog.info("Timestepping complete: stop iteration reached.")
        elif self.time >= self.sim_stop_time:
            ok_flag = False
            if com_sys.myproc == 0:
                mylog.info("Timestepping complete: simulation stop time reached.")
        elif (time.time() - self._start_time) >= self.wall_stop_time:
            ok_flag = False
            if com_sys.myproc == 0:
                mylog.info("Timestepping complete: wall stop time reached.")
        else:
            ok_flag = True

        return ok_flag

    @timer
    def advance(self, data, dt=None):
        if ((self.iteration % self.save_cadence) == 0) or (self.time - self._tlastsnap >= self.max_save_period):
            self.snapshot(data)
        if dt == None:
            dt = self.cfl_dt(data)
        self.do_advance(data,dt)
        mylog.info("step %i" % self.iteration)

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

    def final_stats(self):
        self.timer.print_stats()
        if com_sys.myproc == 0:
            total_wtime = self.timer.timers['advance']
            print "total advance wall time: %10.5e sec" % total_wtime
            print "%10.5e sec/step " %(total_wtime/self.iteration)
            print
            print "Simulation complete. Status: awesome"

    def finalize(self, data):
        """helper function to call at the end of a run to clean up,
        write final output, and print stats.

        """
        self.snapshot(data)
        self.final_stats()

    def cfl_dt(self, data):
        dt = self.CFL * self.RHS.compute_dt(data)

        # only let it increase by 5%
        if dt > 1.05*self.dt_old:
            dt = 1.05 * self.dt_old

        self.dt_old = dt
        mylog.info("dt = %10.5e" % dt)
        return dt


class RKBase(TimeStepBase):
    """Base class for all Runge-Kutta integrators.

    """
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


class RK2mid(RKBase):
    """
    Second-order explicit midpoint Runge-Kutta method, using exponential
    time differencing to handle integrating factors.

    Notes
    -----
    dy/dt = c * y + f(y)

    if c == 0:
        a_n = y_n + dt / 2 * f(y_n)
        y_(n+1) = y_n + dt * f(a_n)
                = a_n + dt * (f(a_n) - f(y_n) / 2)

    else:
        eif = exp(c * dt / 2)
        a_n = y_n * eif + (eif - 1) * f(y_n) / c
        eif = exp(c * dt)
        y_(n+1) = y_n * eif + {[(c * dt - 2) * eif + c * dt + 2] * f(y_n) + [2 * (eif - c * dt - 1)] * f(a_n)} / (c * c * dt)

      0  |  0    0
     1/2 | 1/2   0
    ----------------
         |  0    1

    References
    ----------
    Ashi, H. Numerical Methods for Stiff Systems. University of Nottingham PhD
        (2008). http://etheses.nottingham.ac.uk/663/

    """

    def __init__(self, *arg, **kwargs):

        # Inherited initialization
        TimeStepBase.__init__(self, *arg, **kwargs)

        # Create necessary storage
        self.data2 = self.RHS.create_fields(0.)
        self.deriv1 = self.RHS.create_fields(0.)
        self.deriv2 = self.RHS.create_fields(0.)

        # Import Cython functions
        if self.RHS.ndim == 2:
            from forward_step_cy_2d import euler, etd1, etd2rk2
        else:
            from forward_step_cy_3d import euler, etd1, etd2rk2
        self.euler = euler
        self.etd1 = etd1
        self.etd2rk2 = etd2rk2

    def do_advance(self, data, dt):

        # Integrate in kspace
        s = 'kspace'

        # Place references
        data2 = self.data2
        deriv1 = self.deriv1
        deriv2 = self.deriv2

        # Construct a_n
        self.RHS.RHS(data, deriv1)
        for f in data.fields.keys():
            for c in xrange(data[f].ncomp):
                IF = deriv1[f][c].integrating_factor
                if IF is None:
                    self.euler(data[f][c][s], data2[f][c][s], deriv1[f][c][s], dt / 2.)
                else:
                    self.etd1(data[f][c][s], data2[f][c][s], deriv1[f][c][s], -IF, dt / 2.)
        data2.set_time(data.time + dt / 2.)

        # Complete step
        self.RHS.RHS(data2, deriv2)
        for f in data.fields.keys():
            for c in xrange(data[f].ncomp):
                IF = deriv1[f][c].integrating_factor
                if IF is None:
                    self.euler(data[f][c][s], data[f][c][s], deriv2[f][c][s], dt)
                else:
                    self.etd2rk2(data[f][c][s], data[f][c][s], deriv1[f][c][s], deriv2[f][c][s], -IF, dt)
        data.set_time(data.time + dt)

        # Update integrator stats
        self.time += dt
        self.iteration += 1


class RK2trap(RKBase):
    """
    Second-order explicit trapezoidal Runge-Kutta method, using exponential
    time differencing to handle integrating factors.

    Notes
    -----
    dy/dt = c * y + f(y)

    if c == 0:
        a_n = y_n + dt * f(y_n)
        y_(n+1) = y_n + dt / 2 * (f(a_n) + f(y_n))
                = a_n + dt / 2 * (f(a_n) - f(y_n))
    else:
        eif = exp(c * dt)
        a_n = y_n * eif + (eif - 1) * f(y_n) / c
        y_(n+1) = a_n + (eif - c * dt - 1) * (f(a_n) - f(y_n)) / (c * c * dt)

      0  |  0    0
      1  |  1    0
    ----------------
         | 1/2  1/2

    References
    ----------
    Ashi, H. Numerical Methods for Stiff Systems. University of Nottingham PhD
        (2008). http://etheses.nottingham.ac.uk/663/

    """

    def __init__(self, *arg, **kwargs):

        # Inherited initialization
        TimeStepBase.__init__(self, *arg, **kwargs)

        # Create necessary storage
        self.deriv1 = self.RHS.create_fields(0.)
        self.deriv2 = self.RHS.create_fields(0.)

        # Import Cython functions
        if self.RHS.ndim == 2:
            from forward_step_cy_2d import euler, etd1, etd2rk1
        else:
            from forward_step_cy_3d import euler, etd1, etd2rk1
        self.euler = euler
        self.etd1 = etd1
        self.etd2rk1 = etd2rk1

    def do_advance(self, data, dt):

        # Integrate in kspace
        s = 'kspace'

        # Place references
        deriv1 = self.deriv1
        deriv2 = self.deriv2

        # Construct a_n
        self.RHS.RHS(data, deriv1)
        for f in data.fields.keys():
            for c in xrange(data[f].ncomp):
                IF = deriv1[f][c].integrating_factor
                if IF is None:
                    self.euler(data[f][c][s], data[f][c][s], deriv1[f][c][s], dt)
                else:
                    self.etd1(data[f][c][s], data[f][c][s], deriv1[f][c][s], -IF, dt)
        data.set_time(data.time + dt)

        # Complete step
        self.RHS.RHS(data, deriv2)
        for f in data.fields.keys():
            for c in xrange(data[f].ncomp):
                IF = deriv1[f][c].integrating_factor
                if IF is None:
                    self.euler(data[f][c][s], data[f][c][s], deriv2[f][c][s] - deriv1[f][c][s], dt / 2.)
                else:
                    self.etd2rk1(data[f][c][s], data[f][c][s], deriv1[f][c][s], deriv2[f][c][s], -IF, dt)

        # Update integrator stats
        self.time += dt
        self.iteration += 1


class RK4(RKBase):
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

        # Inherited initialization
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
        self.RHS.RHS(data, self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] = self.temp_data[fname][cindex]['kspace'] / 6.

                # Retrieve initial integrating factors for the complete step
                if comp._static_k or (self.temp_data[fname][cindex].integrating_factor is None):
                    comp.integrating_factor = self.temp_data[fname][cindex].integrating_factor
                else:
                    comp.integrating_factor = self.temp_data[fname][cindex].integrating_factor.copy()

        # Construct k2
        self.forward_step(data, self.temp_data, self.temp_data, dt / 2.)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt / 2. * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 3.

        self.RHS.RHS(self.temp_data, self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += self.temp_data[fname][cindex]['kspace'] / 3.

        # Construct k3
        self.forward_step(data, self.temp_data, self.temp_data, dt / 2.)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt / 2. * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 3.

        self.RHS.RHS(self.temp_data, self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += self.temp_data[fname][cindex]['kspace'] / 3.

        # Construct k4
        self.forward_step(data, self.temp_data, self.temp_data, dt)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a.RHS(a.value)
            a_final_dt += a.RHS(a.value) / 6.

        self.RHS.RHS(self.temp_data, self.temp_data)
        for fname, field in self.total_deriv:
            for cindex, comp in field:
                comp['kspace'] += self.temp_data[fname][cindex]['kspace'] / 6.

        # Complete step
        self.forward_step(data, self.total_deriv, data, dt)
        for a in self.RHS.aux_eqns.values():
            a.value = a_old + dt * a_final_dt

        # Update integrator stats
        self.time += dt
        self.iteration += 1


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
        self.iteration += 1

