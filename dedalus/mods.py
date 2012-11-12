"""Conveninece wrapper for all tools necessary for a simulation.

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

import dedalus.startup_tasks as __startup_tasks

from dedalus.config import decfg

import signal

from dedalus.utils.logger import mylog

from dedalus.analysis.api import \
    AnalysisSet, \
    VolumeAverageSet, \
    Snapshot, \
    TrackMode, \
    VolumeAverage, \
    PowerSpectrum

from dedalus.data_objects.api import \
    FourierRepresentation, \
    FourierShearRepresentation

from dedalus.init_cond.api import \
    taylor_green, \
    sin_k, \
    cos_k, \
    turb,  \
    turb_new, \
    mcwilliams_spec, \
    MIT_vortices, \
    vorticity_wave, \
    alfven, \
    kida_vortex

from dedalus.physics.api import \
    IncompressibleHydro, \
    BoussinesqHydro, \
    IncompressibleMHD, \
    ShearIncompressibleMHD, \
    LinearCollisionlessCosmology, \
    CollisionlessCosmology, \
    LinearBaryonCDMCosmology, \
    BaryonCDMCosmology, \
    LinearCollisionlessCosmology,\
    CollisionlessCosmology

from dedalus.time_stepping.api import \
    RK2mid,\
    RK4, \
    CrankNicholsonVisc

from dedalus.utils.api import \
    restart, \
    Timer, \
    integrate_quad, \
    integrate_simp, \
    interp_linear, \
    find_zero, \
    load_all, \
    com_sys

from funcs import signal_print_traceback
try:
    signal.signal(signal.SIGHUP, signal_print_traceback)
except ValueError:  # Not in main thread
    pass

# load plugins
# WARNING WARNING WARNING: this is *really* dangerous.
# be careful out there, good buddies.

if decfg.getboolean("utils","loadplugins"):
    my_plugin_name = decfg.get("utils","pluginfilename")
    # We assume that it is with respect to the $HOME/.dedalus directory
    if os.path.isfile(my_plugin_name):
        _fn = my_plugin_name
    else:
        _fn = os.path.expanduser("~/.dedalus/%s" % my_plugin_name)
    if os.path.isfile(_fn):
        mylog.info("Loading plugins from %s", _fn)
        execfile(_fn)

