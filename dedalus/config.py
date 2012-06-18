"""Configuration file parser.

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

decfgDefaults = {}

if os.path.exists(os.path.expanduser("~/.dedalus/config")):
    decfg = ConfigParser.ConfigParser(dedaluscfgDefaults)
    decfg.read(['dedalus.cfg', os.path.expanduser('~/.dedalus/config')])
else:
    decfg = ConfigParser.ConfigParser(decfgDefaults)
    decfg.read(['dedalus.cfg'])
if not decfg.has_section("dedalus"):
    decfg.add_section("dedalus")

