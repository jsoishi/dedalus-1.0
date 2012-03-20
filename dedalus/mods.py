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


from dedalus.analysis.api import \
    AnalysisSet, \
    VolumeAverageSet

from dedalus.data_objects.api import \
    FourierRepresentation, \
    FourierShearRepresentation, \
    ParallelFourierRepresentation, \
    ParallelFourierShearRepresentation

from dedalus.init_cond.api import \
    taylor_green, \
    sin_x, \
    sin_y, \
    turb,  \
    mcwilliams_spec, \
    MIT_vortices, \
    shearing_wave, \
    collisionless_cosmo_fields, \
    cosmo_fields, \
    cosmo_spectra, \
    alfven, \
    zeldovich, \
    kida_vortex

from dedalus.physics.api import \
    Hydro, \
    ShearHydro, \
    MHD, \
    ShearMHD, \
    LinearCollisionlessCosmology, \
    CollisionlessCosmology, \
    LinearBaryonCDMCosmology, \
    BaryonCDMCosmology, \
    LinearCollisionlessCosmology,\
    CollisionlessCosmology

from dedalus.time_stepping.api import \
    RK2simple,\
    RK2simplevisc, \
    CrankNicholsonVisc, \
    RK4simplevisc

from dedalus.utils.api import \
    restart, \
    Timer, \
    integrate_quad, \
    integrate_simp, \
    interp_linear, \
    find_zero, \
    setup_parallel_objs, \
    load_all, \
    com_sys
