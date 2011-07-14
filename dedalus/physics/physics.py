"""Physics class. This defines fields, and provides a right hand side
to time integrators.

Authors: J. S. Oishi <jsoishi@gmail.com>
         K. J. Burns <kburns@berkeley.edu>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi, K. J. Burns.  All Rights Reserved.

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
from dedalus.data_objects.api import create_field_classes, zero_nyquist, AuxEquation, StateData
from dedalus.utils.api import friedmann
from dedalus.funcs import insert_ipython

class Physics(object):
    """
    This is a base class for a physics object. It needs to provide
    a right hand side for a time integration scheme.
    """
    
    def __init__(self, shape, representation):
        self.shape = shape
        self.ndim = len(self.shape)
        self.dims = xrange(self.ndim)
        self._representation = representation
        self._field_classes = create_field_classes(
                self._representation, self.shape, self.__class__.__name__)
        self.parameters = {}
        self.aux_eqns = {}
    
    def __getitem__(self, item):
         value = self.parameters.get(item, None)
         if value is None:
              raise KeyError
         return value

    def create_fields(self, t, field_list=None):        
        if field_list == None:
            field_list = self.fields
        return StateData(t, self._field_classes, field_list)    

    def create_dealias_field(self, t, fields=None):
        """data object to implement Orszag 3/2 rule for non-linear
        terms.

        """
        name = "%sDealiasData" % self.__class__.__name__
        shape = [3*d/2 for d in self.shape]
        data_class = create_data(self._representation, shape, name)

        if fields == None:
            return data_class(self.fields, t)
        else:
            return data_class(fields, t)

    def _setup_parameters(self, params):
        for k,v in params.iteritems():
            self.parameters[k] = v

    def _setup_aux_fields(self, t, aux_field_list=None):
        self.aux_fields = self.create_fields(t, aux_field_list)

    def _setup_aux_eqns(self, aux_eqns, RHS, ics):
        """ create auxiliary ODEs to be solved alongside the spatial gradients.

        inputs
        ------
        aux_eqns -- a list of names for the equations
        RHS -- 

        """
        for f, r, ic in zip(aux_eqns, RHS, ics):
            self.aux_eqns[f] = AuxEquation(r, ic)

    def RHS(self):
        pass
        
    def gradX(self, X, output):
        """
        Compute Jacobian: gradX[N * i + j] = dX_i/dx_j
        
        Inputs:
            X           Input VectorField object
            output      Output TensorField object

        """
        
        N = self.ndim

        # Construct Jacobian
        for i in self.dims:
            for j in self.dims:
                output[N * i + j]['kspace'] = X[i].deriv(self._trans[j]) 
                
    def XgradY(self, X, Y, gradY, output, dealias='2/3', compute_gradY=True):
        """
        Calculate "X dot (grad X)" term, with dealiasing options.
        
        Inputs:
            X           Input VectorField object
            Y           Input VectorField object
            gradY       TensorField object to hold gradY
            output      Output VectorField object
            
        Dealiasing options: 
            None        No dealiasing
            '2/3'       Dealias by keeping lower 2/3 modes, and zeroing others
            '3/2'       Dealias by extending to 3/2 larger temp fields
            
        Keywords:
            compute_gradY   Set to False if gradY has been computed
        
        """
        
        if dealias not in [None, '2/3', '3/2']:
            raise ValueError('Dealising method not implemented.')

        if dealias == '3/2':
            # Uses temporary dealias fields with 3/2 as many points 
            # as the shape of the data object.
            
            # ****** THIS HAS NOT BEEN UPDATED TO WORK WITH NEW HANDLING ******

            d = data['ux']
            self.gradu(data)
            gradu = self.aux_fields['gradu']
            ugradu = self.aux_fields['ugradu']
            trans = {0: 'ux', 1: 'uy', 2: 'uz'}
            self.q.zero_all()
            for i,f in enumerate(self.fields):
                b = [i * self.ndim + j for j in self.dims]
                tmp = na.zeros_like(self.q['ugu'].data)
                for ii,j, in enumerate(b):
                    self.q['u'].data[:]= 0+0j
                    self.q['u']._curr_space = 'kspace'
                    self.q['gu'].data[:] = 0+0j
                    self.q['gu']._curr_space = 'kspace'
                    zero_nyquist(data[trans[ii]]['kspace'])
                    self.q['u'] = na.fft.fftshift(data[trans[ii]]['kspace'])
                    self.q['u'] = na.fft.fftshift(self.q['u']['kspace'])
                    self.q['gu'] = na.fft.fftshift(gradu[j]['kspace'])
                    self.q['gu'] = na.fft.fftshift(self.q['gu']['kspace'])
                    tmp += self.q['u']['xspace'] * self.q['gu']['xspace']
                tmp.imag = 0.
                self.q['ugu'] = tmp
                self.q['ugu']._curr_space = 'xspace'
                ugradu[f] = self.q['ugu']['kspace']
                tmp *= 0+0j

        else:
            N = self.ndim
        
            # Perform gradY calculation
            if compute_gradY:
                self.gradX(Y, gradY)

            # Setup temporary data container and dealias mask
            sampledata = X[0]
            tmp = na.zeros_like(sampledata.data)
            
            if dealias == '2/3': 
                # Orszag 2/3 dealias mask (picks out coefficients to zero)    
                dmask = ((na.abs(sampledata.k['x']) > 2/3. *self.shape[0]/2.) | 
                         (na.abs(sampledata.k['y']) > 2/3. * self.shape[1]/2.))
            
            # Construct XgradX **************** Proper dealiasing?
            for i in self.dims:
                for j in xrange(N):
                    tmp += X[j]['xspace'] * gradY[N * i + j]['xspace']

                output[i]['xspace'] = tmp.real
                if dealias == '2/3':
                    output[i]['kspace'] # dummy call to switch spaces
                    output[i]['kspace'][dmask] = 0.
                    
                tmp *= 0+0j
                
    def XcrossY(self, X, Y, output, space):
        """
        Calculate X cross Y.
        
        Inputs:
            X           Input VectorField object
            Y           Input VectorField object
            output      Output Vector/ScalarField object (3D/2D)
            space       Space for cross product
        
        Note: 
            2D inputs require scalar output
            3D inputs require vector output

        """            

        N = X.ndim
            
        # Place references
        X0 = X[0][space]
        X1 = X[1][space]
        if N == 3: X2 = X[2][space]
            
        Y0 = Y[0][space]
        Y1 = Y[1][space]
        if N == 3: Y2 = Y[2][space]

        # Calculate cross product, scalar if N == 2
        if N == 3:
            output[0][space] = X1 * Y2 - X2 * Y1
            output[1][space] = X2 * Y0 - X0 * Y2
            output[2][space] = X0 * Y1 - X1 * Y0
        else:
            output[space] = X0 * Y1 - X1 * Y0
            
    def curlX(self, X, output):
        """
        Return list of components of curl X.
        
        Inputs:
            X           Input VectorField object
            output      Output Vector/ScalarField object (3D/2D)
        
        Note: 
            2D input requires scalar output
            3D input requires vector output
            
        """
        
        N = X.ndim

        # Calculate curl, scalar if N == 2
        if N == 3:
            output[0]['kspace'] = X[2].deriv(self._trans[1]) - X[1].deriv(self._trans[2])
            output[1]['kspace'] = X[0].deriv(self._trans[2]) - X[2].deriv(self._trans[0])
            output[2]['kspace'] = X[1].deriv(self._trans[0]) - X[0].deriv(self._trans[1])
        else:
            output['kspace'] = X[1].deriv(self._trans[0]) - X[0].deriv(self._trans[1])    
        
class Hydro(Physics):
    """Incompressible hydrodynamics."""
    
    def __init__(self, *args):
        Physics.__init__(self, *args)
        
        # Setup data fields
        self.fields = [('u', 'vector')]
        self._aux_fields = [('pressure', 'vector'),
                            ('gradu', 'tensor'),
                            ('ugradu', 'vector')]
        
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'nu': 0., 'rho0': 1.}

        #self.q = self.create_dealias_field(0.,['u','gu','ugu'])
        self._setup_aux_fields(0., self._aux_fields)
        self._setup_parameters(params)
        self._RHS = self.create_fields(0.)

    def RHS(self, data):
        """
        Compute right hand side of fluid equations, populating self._RHS with
        the time derivatives of the fields.

        u_t + nu k^2 u = -ugradu - i k p / rho0

        """
        
        # Compute terms
        self.XgradY(data['u'], data['u'], self.aux_fields['gradu'],
                    self.aux_fields['ugradu'], dealias='2/3')
        self.pressure(data)
        
        # Place references
        ugradu = self.aux_fields['ugradu']
        pressure = self.aux_fields['pressure']
        
        # Construct time derivatives
        for i in self.dims:
            self._RHS['u'][i]['kspace'] = -ugradu[i]['kspace'] - pressure[i]['kspace']
            
        # Set integrating factors
        self._RHS['u'].integrating_factor = self.parameters['nu'] * self._RHS['u']['x'].k2()
            
        self._RHS.time = data.time        
        return self._RHS

    def pressure(self, data):
        """
        Compute pressure term for ufields: i k p / rho0
        
        p / rho0 = i k * ugradu / k^2     
        ==> pressure term = - k (k * ugradu) / k^2
        
        """
        
        # Place references
        ugradu = self.aux_fields['ugradu']
        pressure = self.aux_fields['pressure']
        
        # Setup temporary data container
        sampledata = data['u']['x']
        tmp = na.zeros_like(sampledata.data)
        k2 = sampledata.k2(no_zero=True)
        
        # Construct k * ugradu
        for i in self.dims:
            tmp += data['u'][i].k[self._trans[i]] * ugradu[i]['kspace']

        # Construct full term
        for i in self.dims:            
            pressure[i]['kspace'] = -data['u'][i].k[self._trans[i]] * tmp / k2
            zero_nyquist(pressure[i].data)
 
class MHD(Hydro):
    """Incompressible magnetohydrodynamics."""
    
    def __init__(self, *args):
        Physics.__init__(self, *args)
        
        # Setup data fields
        self.fields = [('u', 'vector'),
                       ('B', 'vector')]
        self._aux_fields = [('Ptotal', 'vector'),
                            ('gradu', 'tensor'),
                            ('ugradu', 'vector'),
                            ('gradB', 'tensor'),
                            ('BgradB', 'vector'),
                            ('ugradB', 'vector'),
                            ('Bgradu', 'vector')]
        
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'nu': 0., 'rho0': 1., 'eta': 0.}

        #self.q = self.create_dealias_field(0.,['u','gu','ugu'])
        self._setup_aux_fields(0., self._aux_fields)
        self._setup_parameters(params)
        self._RHS = self.create_fields(0.)

    def RHS(self, data):
        """
        Compute right hand side of fluid equations, populating self._RHS with
        the time derivatives of the fields.
        
        u_t + nu k^2 u = -ugradu + BgradB / (4 pi rho0) - i k Ptot / rho0
        
        B_t + eta k^2 B = Bgradu - ugradB

        """
        
        # Place references
        u = data['u']
        B = data['B']
        gradu = self.aux_fields['gradu']
        gradB = self.aux_fields['gradB']
        ugradu = self.aux_fields['ugradu']
        BgradB = self.aux_fields['BgradB']
        ugradB = self.aux_fields['ugradB']
        Bgradu = self.aux_fields['Bgradu']
        Ptotal = self.aux_fields['Ptotal']
        pr4 = 4 * na.pi * self.parameters['rho0']
        
        # Compute terms
        self.XgradY(u, u, gradu, ugradu, dealias='2/3')
        self.XgradY(B, B, gradB, BgradB, dealias='2/3')
        self.XgradY(u, B, gradB, ugradB, dealias='2/3', compute_gradY=False)
        self.XgradY(B, u, gradu, Bgradu, dealias='2/3', compute_gradY=False)
        self.total_pressure(data)
        
        # Construct time derivatives
        for i in self.dims:
            self._RHS['u'][i]['kspace'] = (-ugradu[i]['kspace'] + 
                                           BgradB[i]['kspace'] / pr4 -
                                           Ptotal[i]['kspace'])
                                           
            self._RHS['B'][i]['kspace'] = Bgradu[i]['kspace'] - ugradB[i]['kspace']

        # Set integrating factors
        self._RHS['u'].integrating_factor = self.parameters['nu'] * self._RHS['u']['x'].k2()
        self._RHS['B'].integrating_factor = self.parameters['eta'] * self._RHS['B']['x'].k2()

        self._RHS.time = data.time        
        return self._RHS
        
    def total_pressure(self, data):
        """
        Compute total pressure term (including magnetic): i k Ptot / rho0
        
        Ptot / rho0 = i k * ugradu / k^2 - i k * BgradB / (4 pi rho0 k^2)  
        ==> pressure term = - k (k * ugradu - k * BgradB / (4 pi rho0)) / k^2
        
        """

        # Place references
        ugradu = self.aux_fields['ugradu']
        BgradB = self.aux_fields['BgradB']
        Ptotal = self.aux_fields['Ptotal']
        pr4 = 4 * na.pi * self.parameters['rho0']

        # Setup temporary data container
        sampledata = data['u']['x']
        tmp = na.zeros_like(sampledata.data)
        k2 = sampledata.k2(no_zero=True)

        # Construct k * ugradu - k * BgradB / (4 pi rho0)
        for i in self.dims:
            tmp += data['u'][i].k[self._trans[i]] * ugradu[i]['kspace']
            tmp -= data['u'][i].k[self._trans[i]] * BgradB[i]['kspace'] / pr4
        
        # Construct full term
        for i in self.dims:            
            Ptotal[i]['kspace'] = -data['u'][i].k[self._trans[i]] * tmp / k2
            zero_nyquist(Ptotal[i].data)

    
            
class LinearCollisionlessCosmology(Physics):
    """This class implements linear, collisionless cosmology. 

    parameters
    ----------
    Omega_r - energy density in radiation
    Omega_m - energy density in matter
    Omega_l - energy density in dark energy (cosmological constant)
    H0      - hubble constant now (100 if all units are in km/s/Mpc)
    solves
    ------
    d_t d = - div(v_)
    d_t v = -grad(Phi) - H v
    laplacian(Phi) = 3 H^2 d / 2

    """
    def __init__(self,*args):
        Physics.__init__(self, *args)
        self.fields = ['delta','ux','uy']
        self._aux_fields = ['gradphi']
        aux_types = [None]
        if self.ndim == 3:
             self.fields.append('uz')
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'Omega_r': 0.,
                  'Omega_m': 0.,
                  'Omega_l': 0.,
                  'H0': 100.}
        self._setup_parameters(params)
        self._setup_aux_fields(0., self._aux_fields,aux_types)
        aux_eqn_rhs = lambda a: a*friedmann(a,self.parameters['H0'],self.parameters['Omega_r'], self.parameters['Omega_m'], self.parameters['Omega_l'])
        
        self._setup_aux_eqns(['H'],[aux_eqn_rhs], [1e-5])

        self._RHS = self.create_fields(0.)

    def RHS(self, data):
        self.density_RHS(data)
        
        return self._RHS

    def density_RHS(self, data):
        pass

    def vel_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']
        for i,f in enumerate(self.fields[1:]):
            self._RHS[f] = -gradphi[f]['kspace'] - self.aux_eqns['H'] * data[f]['kspace']
        
    def grad_phi(self, data):
        gradphi = self.aux_fields['gradphi']
        tmp = -3./2. * self.parameters['H0'] * data['delta']['kspace']/data['delta'].k2(no_zero=True)
        
        for i,f in enumerate(self.fields[1:]):
            gradphi[f] = data[f].k(self._trans(i)) * tmp

if __name__ == "__main__":
    import pylab as P
    from fourier_data import FourierData
    from init_cond import taylor_green
    a = Hydro((100,100),FourierData)
    data = a.create_fields(0.)
    taylor_green(data['ux'],data['uy'])
    vgv = a.ugradu(data)
    #test = a.pressure(data,vgv)
    test = a.RHS(data)

    for i,f in enumerate(a.fields):
        print test[f]._curr_space
        P.subplot(1,2,i+1)
        P.imshow(test[f]['xspace'].real)
        tmp =test[f]['xspace'].imag
        print "%s (min, max) = (%10.5e, %10.5e)" % (f, tmp.min(), tmp.max())
        P.colorbar()

    P.show()
