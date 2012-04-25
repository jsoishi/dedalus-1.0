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
import logging.handlers as handlers
from dedalus.utils.parallelism import com_sys
#from yt.config import ytcfg

class SingleLevelFilter(logging.Filter):
    """grabbed from http://stackoverflow.com/questions/1383254/logging-streamhandler-and-standard-streams

    """
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)

level = logging.DEBUG

ufstring = "%(asctime)s %(name)-3s: [%(levelname)-9s] %(proc)i %(message)s"
formatter = logging.Formatter(ufstring)

rootLogger = logging.getLogger()

ti = {'proc': com_sys.myproc}
dedaluslog = logging.getLogger("Dedalus")
mylog = logging.LoggerAdapter(dedaluslog, ti)

dedaluslog.setLevel(logging.DEBUG)

f1 = SingleLevelFilter(logging.DEBUG, False)
debug_handler = logging.StreamHandler(sys.stderr)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(f1)
debug_handler.setFormatter(formatter)
dedaluslog.addHandler(debug_handler)

if com_sys.myproc == 0:
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logging.root.addHandler(info_handler)

def disable_stream_logging():
    # We just remove the root logger's handlers
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            rootLogger.removeHandler(handler)

mylog.debug("Set log level to %s", level)
