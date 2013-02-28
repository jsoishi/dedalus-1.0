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
from dedalus.utils.parallelism import com_sys, reduce_sum, reduce_mean, reduce_max
from dedalus.utils.logger import mylog
import numpy as na

class VolumeAverageSet(object):

    # Dictionary of registered tasks
    known_analysis = {}

    def __init__(self, data, filename='time_series.dat'):
        self.data = data
        self.filename = filename
        self.tasks = []
        self.scratch = self.data.clone()
        self.scratch.add_field('scalar', 'ScalarField')
        self.scratch.add_field('vector', 'VectorField')
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
                retval = f(self.data, self.scratch)
            else:
                retval = f(self.data, self.scratch, **kwargs)
            if com_sys.myproc == 0:
                line.append(fmt % retval)
        if com_sys.myproc == 0:
            self.outfile.write("\t".join(line)+"\n")
            self.outfile.flush()

    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func

def volume_average(data, kdict=None, space='kspace', reduce_all=False):
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

        return reduce_sum(local_sum,reduce_all=reduce_all)
    elif space == 'xspace':
        if type(data) == na.ndarray:
            data_values = data
        else:
            data_values = data['xspace']
        return reduce_mean(data_values)
    else:
        raise ValueError("volume_average: must be either xspace or kspace")

@VolumeAverageSet.register_task
def ekin(data, scratch, space='kspace'):
    scratch['scalar'].zero_all()
    for i in xrange(data['u'].ncomp):
        if space == 'kspace':
            scratch['scalar']['kspace'] += 0.5 * na.abs(data['u'][i]['kspace']) ** 2
        else:
            scratch['scalar']['xspace'] += 0.5 * na.abs(data['u'][i]['xspace']) ** 2


    return volume_average(scratch['scalar'], space=space)

@VolumeAverageSet.register_task
def comp_mean(data, scratch, fname, cindex):
    return volume_average(data[fname][cindex]['xspace'], space='xspace')

@VolumeAverageSet.register_task
def ux2(data, scratch, space='kspace'):
    return volume_average(data['u']['x']['kspace']*data['u']['x']['kspace'].conj(), kdict=data['u']['x'].k)

@VolumeAverageSet.register_task
def uy2(data, scratch, space='kspace'):
    return volume_average(data['u']['y']['kspace']*data['u']['y']['kspace'].conj(), kdict=data['u']['y'].k)

@VolumeAverageSet.register_task
def uz2(data, scratch, space='kspace'):
    return volume_average(data['u']['z']['kspace']*data['u']['z']['kspace'].conj(), kdict=data['u']['z'].k)

@VolumeAverageSet.register_task
def uxm(data, scratch, space='kspace'):
    return volume_average(data['u']['x'], space='xspace')

@VolumeAverageSet.register_task
def uym(data, scratch, space='kspace'):
    return volume_average(data['u']['y'], space='xspace')

@VolumeAverageSet.register_task
def uzm(data, scratch, space='kspace'):
    return volume_average(data['u']['z'], space='xspace')

@VolumeAverageSet.register_task
def temp2(data, scratch, space='kspace'):
    """mean square temperature

    """
    return volume_average(data['T']['kspace']*data['T']['kspace'].conj(), kdict=data['T'].k, space=space)

@VolumeAverageSet.register_task
def emag(data, scratch, space='kspace'):
    for i in xrange(data['B'].ncomp):
        if space == 'kspace':
            scratch['scalar']['kspace'] += 0.5 * na.abs(data['B'][i]['kspace']) ** 2
        else:
            scratch['scalar']['xspace'] += 0.5 * na.abs(data['B'][i]['xspace']) ** 2

    return volume_average(scratch['scalar'], space=space)

@VolumeAverageSet.register_task
def enstrophy(data, scratch, space='kspace'):
    """compute enstrophy

    HARDCODED FOR 2D CARTESIAN ONLY!

    """
    #aux = data.__class__(['vortz'],data.time)
    scratch['scalar']['kspace'] = 0.5*na.abs((data['u']['y'].deriv('x') - data['u']['x'].deriv('y'))**2)

    return volume_average(scratch['scalar'],space='kspace')

@VolumeAverageSet.register_task
def vort_cenk(data, scratch):
    """Centroid wavenumber from McWilliams 1990.

    """
    k2 = data['u']['x'].k2(no_zero=True)
    en = na.zeros(data['u']['x'].data.shape)
    for i in xrange(data['u'].ndim):
        en += 0.5*(data['u'][i]['kspace']*data['u'][i]['kspace'].conj()).real
    en[0,0] = 0. # compensate for the fact that k2 has mean mode = 1
    return (k2**1.5*en).sum()/(k2*en).sum()

@VolumeAverageSet.register_task
def ux_imag(data, scratch):
    return data['u']['x']['xspace'].imag.mean()

@VolumeAverageSet.register_task
def uy_imag(data, scratch):
    return data['u']['y']['xspace'].imag.mean()

@VolumeAverageSet.register_task
def ux_imag_max(data, scratch):
    return data['u']['x']['xspace'].imag.max()

@VolumeAverageSet.register_task
def uy_imag_max(data, scratch):
    return data['u']['y']['xspace'].imag.max()

@VolumeAverageSet.register_task
def energy_dissipation(data, scratch):
    """energy dissipation rate: 2 nu vorticity**2

    """
    scratch['scalar']['kspace'] = 0.5 * ((na.abs(data['u']['z'].deriv('y') - data['u']['y'].deriv('z')))**2 \
        + (na.abs(data['u']['x'].deriv('z') - data['u']['z'].deriv('x')))**2 \
        + (na.abs(data['u']['y'].deriv('x') - data['u']['x'].deriv('y')))**2)

    en_dis = 2*data.parameters['nu']*(scratch['scalar']['kspace']).real

    return volume_average(en_dis,kdict=scratch['scalar'].k)

@VolumeAverageSet.register_task
def thermal_energy_dissipation(data, scratch):
    """energy dissipation rate: kappa <d_i T>**2

    """
    scratch['scalar']['kspace'] = data.parameters['kappa'] * ((na.abs(data['T'].deriv('x')))**2
        + (na.abs(data['T'].deriv('y')))**2 \
        + (na.abs(data['T'].deriv('z')))**2)

    return volume_average(scratch['scalar'])

@VolumeAverageSet.register_task
def divergence(data, scratch):
    scratch['scalar']['kspace'] = data['u']['x'].deriv('x') \
        + data['u']['y'].deriv('y')
    if data.ndim == 3:
        scratch['scalar']['kspace'] += data['u']['z'].deriv('z')

    return volume_average(scratch['scalar'])

@VolumeAverageSet.register_task
def divergence_sum(data, scratch):

    scratch['scalar'].zero()
    for cindex, comp in data['u']:
        cname = data['u'].ctrans[cindex]
        scratch['scalar']['kspace'] += comp.deriv(cname)

    return abs(scratch['scalar']['kspace']).sum()

@VolumeAverageSet.register_task
def div_norm(data, scratch):
    scratch['scalar'].zero()
    for cindex, comp in data['u']:
        cname = data['u'].ctrans[cindex]
        scratch['scalar']['kspace'] += comp.deriv(cname)
    dx = (data['u'][0].dx()).max()
    return reduce_max(dx*scratch['scalar']['xspace']/data['u'].l2norm())
