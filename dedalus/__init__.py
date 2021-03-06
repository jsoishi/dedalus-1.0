"""
Dedalus is a pseudospectral solver for fluid equations. Its primary
applications are in Astrophysics and Cosmology. Written primarily in
python, and making use of the FFTW libraries, Dedalus aims to be a
simple, fast, and elegant hydrodynamic and magnetohydrodynamic code.

Dedalus is divided into several subpackages

analysis
--------
tools for inline analysis and plotting go here.

data_objects
------------
The basic data objects of Dedalus define both the spatial data layout
*and* the spatial derivative operator on that data. Currently, only
periodic domains represented by Fourier basis functions are supported.

init_cond
---------
Various initial conditions generators.

physics
-------
Each physics module defines the terms to be fed to the right hand side
(RHS) of a time integrator. These modules create the equations solved
by the code.

samples
-------
Sample scripts and test problems live here.

time_stepping
-------------
Various time integrators derived from a base class that provides a
number of nice features for PDEs.

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




