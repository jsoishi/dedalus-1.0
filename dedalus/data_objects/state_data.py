"""The Main Dedalus data object. Has a dictionary of fields, each of
which can be a Tensor, Vector, or Scalar object.

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

class StateData(object):
    """the object containing all relevant data for the state of the
    system. composed of vector and scalar fields, each of which in
    turn is determined at start up with 1,2, or 3 componentes, named
    according to the coordinate system in use (xyz/rthetaphi/rphiz/etc
    etc etc)...or 0,1,2
    """
    def __init__(self, time, TensorClass, VectorClass, ScalarClass):
        self.time = time
        self.fields = {}
        self._field_classes = {'tensor': TensorClass,
                                'vector': VectorClass, 
                                'scalar': ScalarClass}

    def __getitem__(self, item):
        return self.fields[item]

    def add_field(self, field, field_type):
        """add a new field. There is a SIGNIFICANT performace penalty
        for doing this (creating the FFTW plan), so make sure it does
        not happen inside any loops you care about performance on....

        """
        if field not in self.fields.keys():
            self.fields[field] = self._field_classes[field_type]()

    def snapshot(self, nsnap):
        """NEEDS TO BE UPDATED FOR NEW FIELD TYPES

        """
        pass
        #filename = "snap_%05i.cpu%04i" % (nsnap, 0)
        #outfile = h5py.File(filename, mode='w')
        #grp = outfile.create_group('/data')
        #dset = outfile.create_dataset('time',data=self.time)
        #for f in self.fields.keys():
        #    dset = grp.create_dataset(f,data=self[f].data)
        #    dset.attrs["space"] = self[f]._curr_space
        #
        #outfile.close()

