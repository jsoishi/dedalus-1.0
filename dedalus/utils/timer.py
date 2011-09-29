"""Timers to track code performance

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
class Timer(object):
    timers = {}
    def __init__(self):
        """a simple class that builds a dictionary of functions and
        their cumulative times.

        """
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            stop = time.time()
            try:
                self.timers[func.func_name] += (stop-start)
            except KeyError:
                self.timers[func.func_name] = (stop-start)
        return wrapper

    def print_stats(self):
        for k,v in self.timers.iteritems():
            print "%s: %10.5f sec" % (k,v)

if __name__ == "__main__":
    from time import sleep
    times = Timer()

    @times
    def blah(f, g):
        sleep(2)

    blah(1,2)

    times.print_stats()
