"""The Main Dedalus data object. This is dynamically created with a
given representation when the physics class is initialized.

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

def create_data(representation, shape, name):
    new_class = type(name, (BaseData,), {'representation': representation,
                                           'shape': shape})
    return new_class

class BaseData(object):
    def __init__(self, fields, time):
        self.time = time
        self.fields = {}
        for f in fields:
            self.fields[f] = self.representation(self.shape)

    def __getitem__(self, item):
        a = self.fields.get(item, None)
        if a is None:
            raise KeyError
        return a

    def __setitem__(self, item, data):
        """this needs to ensure the pointer for the field's data
        member doesn't change for FFTW. Currently, we do that by
        slicing the entire data array. 
        """
        f = self.fields[item]
        if data.size < f.data.size:
            sli = [slice(i/4+1,i/4+i+1) for i in data.shape]
            f.data[sli] = data
        else:
            sli = [slice(i) for i in f.data.shape]
            f.data[:] = data[sli]

    def add_field(self, field):
        """add a new field. There is a SIGNIFICANT performace penalty
        for doing this (creating the FFTW plan), so make sure it does
        not happen inside any loops you care about performance on....

        """
        if field not in self.fields.keys():
            self.fields[field] = self.representation(self.shape)

    def zero(self, item):
        self.fields[item].data[:] = 0.

    def zero_all(self):
        for f in self.fields.keys():
            self.zero(f)

    def snapshot(self):
        for f in fields:
            pass
