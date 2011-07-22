"""useful functions.

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
import sys
import subprocess
import re

__header = """
== Welcome to the embedded IPython Shell ==

   You are currently inside the function:
     %(fname)s

   Defined in:
     %(filename)s:%(lineno)s
"""

def insert_ipython(num_up=1):
    """
    Placed inside a function, this will insert an IPython interpreter at that
    current location.  This will enabled detailed inspection of the current
    exeuction environment, as well as (optional) modification of that environment.
    *num_up* refers to how many frames of the stack get stripped off, and
    defaults to 1 so that this function itself is stripped off.

    this function from yt (http://yt.enzotools.org/)
    """
    from IPython.Shell import IPShellEmbed
    stack = inspect.stack()
    frame = inspect.stack()[num_up]
    loc = frame[0].f_locals.copy()
    glo = frame[0].f_globals
    dd = dict(fname = frame[3], filename = frame[1],
              lineno = frame[2])
    ipshell = IPShellEmbed()
    ipshell(header = __header % dd,
            local_ns = loc, global_ns = glo)
    del ipshell




def get_mercurial_changeset_id():
    """from Jason F. Harris.

    http://jasonfharris.com/blog/2010/05/versioning-your-application-with-the-mercurial-changeset-hash/

    """
    targetBuildDir = os.getenv("TARGET_BUILD_DIR")
    getChangeset = subprocess.Popen('hg parent --template "{node|short}" --cwd ' + targetBuildDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if (getChangeset.stderr.read() != ""):
        print "Error in obtaining current changeset of the Mercurial repository"
        changeset = None

    changeset = getChangeset.stdout.read()
    if (not re.search("^[0-9a-f]{12}$", changeset)):
        print "Current changeset of the Mercurial repository is malformed"
        changeset = None

    return changeset
