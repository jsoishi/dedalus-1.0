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
from dedalus.utils.parallelism import com_sys, reduce_sum, reduce_mean
from dedalus.utils.logger import mylog
import numpy as na

class VolumeAverageSet(object):

    # Dictionary of registered tasks
    known_analysis = {}
    
    def __init__(self, data, filename='time_series.dat'):
        self.data = data
        self.filename = filename
        self.tasks = []
        if com_sys.myproc == 0:
            self.outfile = open(self.filename,'a')
            self.outfile.write("# Dedalus Volume Average\n")
            self.outfile.write("# Column 0: time\n")

    def add(self, name, fmt, options={}):
        self.tasks.append((self.known_analysis[name], fmt, options))
        if com_sys.myproc == 0:
            self.outfile.write("# Column %i: %s\n" % (len(self.tasks), name))
        
    def run(self):
        if com_sys.myproc == 0:
            line = []
            line.append("%10.5f" % self.data.time)
        for f, fmt, kwargs in self.tasks:
            if len(kwargs) == 0:
                retval = f(self.data)
            else:
                retval = f(self.data, **kwargs)
            if com_sys.myproc == 0:
                line.append(fmt % retval)
        if com_sys.myproc == 0:
            self.outfile.write("\t".join(line)+"\n")
            self.outfile.flush()

    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func

def volume_average(data, kdict=None, space='kspace'):
    """computes a volume average using either kspace or xspace. in
    kspace, must make sure to compute properly for kx = 0 zero plane.

    """
    if space == 'kspace':
        try:
            k = data.k
            data_values = data['kspace']
        except AttributeError:
            k = kdict
            data_values = data
            if k == None:
                raise ValueError("volume_average: data is not a Dedalus representation, so you must pass k vectors via kdict")
        
        if data.ndim == 3:
            if k['x'][...,0] == 0:
                local_sum = 2*data_values[...,1:].sum() + data_values[...,0].sum()
            else:
                local_sum = 2*data_values.sum()
                
        else:
            if k['x'][0, ...] == 0:
                local_sum = 2*data_values[1:, ...].sum() + data_values[0, ...].sum()
            else:
                local_sum = 2*data_values.sum()

        return reduce_sum(local_sum)
    elif space == 'xspace':
        data_values = data['xspace']
        return reduce_mean(data_values)
    else:
        raise ValueError("volume_average: must be either xspace or kspace")
                

       
@VolumeAverageSet.register_task
def ekin(data, space='kspace'):
    aux = data.clone()
    aux.add_field('ekin', 'ScalarField')
    for i in xrange(data['u'].ncomp):
        if space == 'kspace':
            aux['ekin']['kspace'] += 0.5 * na.abs(data['u'][i]['kspace']) ** 2
        else:
            aux['ekin']['xspace'] += 0.5 * na.abs(data['u'][i]['xspace']) ** 2


    return volume_average(aux['ekin'], space=space)

@VolumeAverageSet.register_task
def emag(data, space='kspace'):
    aux = data.clone()
    aux.add_field('emag', 'ScalarField')
    for i in xrange(data['B'].ncomp):
        if space == 'kspace':
            aux['emag']['kspace'] += 0.5 * na.abs(data['B'][i]['kspace']) ** 2
        else:
            aux['emag']['xspace'] += 0.5 * na.abs(data['B'][i]['xspace']) ** 2

    return volume_average(aux['emag'], space=space)
    
@VolumeAverageSet.register_task
def mode_track(data, k=(1, 0, 0)):
    # NOT IMPLEMENTED
    return 0.

@VolumeAverageSet.register_task
def enstrophy(data):
    """compute enstrophy 

    HARDCODED FOR 2D CARTESIAN ONLY!

    """
    #aux = data.__class__(['vortz'],data.time)
    aux = data.clone()
    aux.add_field('vortz', 'ScalarField')
    aux['vortz']['kspace'] = data['u']['y'].deriv('x') - data['u']['x'].deriv('y')
    
    enstrophy = (aux['vortz']['xspace']**2).real
    return enstrophy.mean()

@VolumeAverageSet.register_task
def vort_cenk(data):
    """Centroid wavenumber from McWilliams 1990.

    """
    k2 = data['u']['x'].k2(no_zero=True)
    en = na.zeros(data['u']['x'].data.shape)
    for i in xrange(data['u'].ndim):
        en += 0.5*(data['u'][i]['kspace']*data['u'][i]['kspace'].conj()).real
    en[0,0] = 0. # compensate for the fact that k2 has mean mode = 1
    return (k2**1.5*en).sum()/(k2*en).sum()

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

@VolumeAverageSet.register_task
def energy_dissipation(data):
    """energy dissipation rate: 2 nu vorticity**2

    """
    aux = data.clone()
    aux.add_field('enstr', 'ScalarField')

    aux['enstr']['kspace'] = 0.5 * ((na.abs(data['u']['z'].deriv('y') - data['u']['y'].deriv('z')))**2 \
        + (na.abs(data['u']['x'].deriv('z') - data['u']['z'].deriv('x')))**2 \
        + (na.abs(data['u']['y'].deriv('x') - data['u']['x'].deriv('y')))**2)
    
    en_dis = 2*data.parameters['nu']*(aux['enstr']['kspace']).real

    return volume_average(en_dis,kdict=aux['enstr'].k)

@VolumeAverageSet.register_task
def divergence(data):
    aux = data.clone()
    aux.add_field('div','ScalarField')

    aux['div']['kspace'] = data['u']['x'].deriv('x') \
        + data['u']['y'].deriv('y') \
        + data['u']['z'].deriv('z')

    div = (aux['div']['kspace'].sum()).real
    return reduce_sum(div)
