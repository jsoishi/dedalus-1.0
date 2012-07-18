"""
Logging facility for Dedalus. 

Parts of this file were borrowed from yt. yt is GPLv3 licensed. 

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

import logging, os, sys
from dedalus.config import decfg
from dedalus.utils.parallelism import com_sys

class SingleLevelFilter(logging.Filter):
    """
    Grabbed from http://stackoverflow.com/questions/1383254/logging-streamhandler-and-standard-streams

    """
    
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)

# Format for logging output
ufstring = "%(asctime)s %(name)-3s: [%(levelname)-9s] %(proc)i %(message)s"
formatter = logging.Formatter(ufstring)
extras = {'proc': com_sys.myproc}

# Construct logger adapter
dedaluslog = logging.getLogger("Dedalus")
loglevel = decfg.get('utils', 'loglevel')
dedaluslog.setLevel(getattr(logging, loglevel.upper()))
mylog = logging.LoggerAdapter(dedaluslog, extras)

# Add stderr handler for DEBUG
is_debug_filter = SingleLevelFilter(logging.DEBUG, False)
debug_handler = logging.StreamHandler(sys.stderr)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
debug_handler.addFilter(is_debug_filter)
dedaluslog.addHandler(debug_handler)

# Add stdout handler for INFO+
if com_sys.myproc == 0:
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    dedaluslog.addHandler(info_handler)

