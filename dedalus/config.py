"""
Configuration file parser.

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2012 J. S. Oishi.  All Rights Reserved.

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

import ConfigParser
import os

decfg = ConfigParser.ConfigParser()

# Default values
decfg.add_section('FFT')
decfg.set('FFT', 'method', 'fftw')
decfg.set('FFT', 'dealiasing', '2/3 cython')

decfg.add_section('utils')
decfg.set('utils', 'loglevel', 'debug')

decfg.add_section('analysis')
decfg.set('analysis', 'snapshot_space', 'xspace')
decfg.set('analysis', 'snapshot_axis', 'z')
decfg.set('analysis', 'snapshot_index', 'middle')
decfg.set('analysis', 'snapshot_units', 'True')

# Read user config, local config
decfg.read([os.path.expanduser('~/.dedalus/config'), 'dedalus.cfg'])


