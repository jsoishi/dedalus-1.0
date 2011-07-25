"""VolumeAverageSet defines a set of volume average data to be written
out to a simple text file. It is called by an AnalysisSet task, so
this does not handle cadence.

Author: Matthew Turk <matthewturk@gmail.com
Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 Matthew Turk, J. S. Oishi.  All Rights Reserved.

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

import numpy as na

class VolumeAverageSet(object):

    # Dictionary of registered tasks
    known_analysis = {}
    
    def __init__(self, data, filename='time_series.dat'):
        self.data = data
        self.filename = filename
        self.tasks = []
        self.outfile = open(self.filename,'wa')
        self.outfile.write("# Dedalus Volume Average\n")
        self.outfile.write("# Column 0: time\n")

    def add(self, name, fmt):
        self.tasks.append((self.known_analysis[name], fmt))
        self.outfile.write("# Column %i: %s\n" % (len(self.tasks), name))

    def run(self):
        line = []
        line.append("%10.5f" % self.data.time)
        for f,fmt in self.tasks:
            line.append(fmt % f(self.data))
        self.outfile.write("\t".join(line)+"\n")
        self.outfile.flush()

    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func

@VolumeAverageSet.register_task
def ekin(data):
    en = na.zeros(data['u']['x'].data.shape)
    for i in xrange(data['u'].ndim):
        en += 0.5*(data['u'][i]['kspace']*data['u'][i]['kspace'].conj()).real

    return en.sum()

@VolumeAverageSet.register_task
def enstrophy(data):
    """compute enstrophy 

    HARDCODED FOR 2D CARTESIAN ONLY!

    """
    aux = data.__class__(['vortz'],data.time)
    aux = data.clone()
    aux.add_field('vortz', 'scalar')
    aux['vortz']['kspace'] = data['u']['y'].deriv('x') - data['u']['x'].deriv('y')
    
    enstrophy = (aux['vortz']['xspace']**2).real
    return enstrophy.mean()

@VolumeAverageSet.register_task
def vort_cenk(data):
    """Centroid wavenumber from McWilliams 1990.

    """
    k2 = data['u']['x'].k2()
    en = na.zeros(data['u']['x'].data.shape)
    for i in xrange(data['u'].ndim):
        en += 0.5*(data['u'][i]['kspace']*data['u'][i]['kspace'].conj()).real
    
    return (k2[1:,1:]**1.5*en[1:,1:]).sum()/(k2[1:,1:]*en[1:,1:]).sum()

@VolumeAverageSet.register_task
def ux_imag(data):
    return data['u']['x']['xspace'].imag.mean()

@VolumeAverageSet.register_task
def uy_imag(data):
    return data['u']['y']['xspace'].imag.mean()

@VolumeAverageSet.register_task
def ux_imag_max(data):
    return data['u']['x']['xspace'].imag.max()

@VolumeAverageSet.register_task
def uy_imag_max(data):
    return data['u']['y']['xspace'].imag.max()
