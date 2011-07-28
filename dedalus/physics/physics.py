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
from dedalus.data_objects.api import create_field_classes, AuxEquation, StateData
from dedalus.utils.api import friedmann
from dedalus.funcs import insert_ipython

def _reconstruct_object(*args, **kwargs):
    new_args = [args[1]['shape'], args[1]['_representation'], args[1]['length']]
    obj = args[0](*new_args)
    obj.__dict__.update(args[1])
    return obj

class Physics(object):
    """
    This is a base class for a physics object. It needs to provide
    a right hand side for a time integration scheme.
    """
    
    def __init__(self, shape, representation, length=None):
        self.shape = shape
        self.ndim = len(self.shape)
        if length:
            self.length = length
        else:
            self.length = (2 * na.pi,) * self.ndim
        self.dims = xrange(self.ndim)
        self._representation = representation
        self._field_classes = create_field_classes(
                self._representation, self.shape, self.length)
        self.parameters = {}
        self.aux_eqns = {}
    
    def __getitem__(self, item):
         value = self.parameters.get(item, None)
         if value is None:
              raise KeyError
         return value

    def __reduce__(self):
        savedict = {}
        exclude = ['aux_fields', '_field_classes']
        for k,v in self.__dict__.iteritems():
            if k not in exclude:
                savedict[k] = v
        return (_reconstruct_object, (self.__class__, savedict))

    def create_fields(self, t, field_list=None):        
        if field_list == None:
            field_list = self.fields
        return StateData(t, self.shape, self.length, self._field_classes, 
                         field_list=field_list, params=self.parameters)

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
            X           Input Scalar/VectorField object
            output      Output Vector/TensorField object

        """

        N = self.ndim

        # Construct Jacobian
        for i in xrange(X.ncomp):
            for j in self.dims:
                output[N * i + j]['kspace'] = X[i].deriv(self._trans[j]) 
                
    def divX(self, X, output):
        """
        Compute divergence: divX = dX_i/dx_i
        
        Inputs:
            X           Input VectorField object
            output      Output ScalarField object

        """

        N = self.ndim
        output.zero()

        # Construct divergence
        for i in xrange(X.ncomp):
            output['kspace'] += X[i].deriv(self._trans[i]) 
                
    def XgradY(self, X, Y, gradY, output, compute_gradY=True):
        """
        Calculate X dot (grad Y).
        
        Inputs:
            X           Input VectorField object
            Y           Input Scalar/VectorField object
            gradY       Vector/TensorField object to hold gradY
            output      Output Scalar/VectorField object
            
        Keywords:
            compute_gradY   Set to False if gradY has been computed

        """

        N = self.ndim
    
        # Perform gradY calculation
        if compute_gradY:
            self.gradX(Y, gradY)

        # Setup temporary data container
        sampledata = X[0]
        tmp = na.zeros_like(sampledata.data)
        
        # Construct XgradY
        for i in xrange(Y.ncomp):
            for j in self.dims:
                tmp += X[j]['xspace'] * gradY[N * i + j]['xspace']
            output[i]['xspace'] = tmp.real                    
            tmp *= 0+0j
            
    def XlistgradY(self, Xlist, Y, tmp, outlist):
        """
        Calculate X dot (grad Y) for X in Xlist.
        This is a low-memory alternative to XgradY (never stores a full gradY tensor).
        
        Inputs:
            Xlist       List of input VectorField objects
            Y           Input Scalar/VectorField object
            tmp         ScalarField object for use in internal calculations
            outlist     List of output Scalar/VectorField object

        """

        N = self.ndim

        # Zero all output fields
        for outfield in outlist:
            outfield.zero_all()

        for i in xrange(Y.ncomp):
            for j in self.dims:
                # Compute dY_i/dx_j
                tmp['kspace'] = Y[i].deriv(self._trans[j])
                
                # Add term to each output
                for k,X in enumerate(Xlist):
                    outlist[k][i]['xspace'] += (X[j]['xspace'] * tmp['xspace']).real
     
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
            
    def XdotY(self, X, Y, output, space):
        """
        Calculate X dot Y.
        
        Inputs:
            X           Input VectorField object
            Y           Input VectorField object
            output      Output ScalarField object
            space       Space for dot product

        """            

        if X.ncomp != Y.ncomp: raise ValueError('Vectors not the same size')

        output.zero()  
        for i in xrange(X.ncomp):
            output[space] += X[i][space] * Y[i][space]
            
    def curlX(self, X, output):
        """
        Calculate curl of X.
        
        Inputs:
            X           Input Scalar/Vector/VectorField object
            output      Output Vector/Scalar/VectorField object (2D/2D/3D)
        
        Note: 
            2D: Vector input requires Scalar output
                Scalar input requires Vector output
            3D: Vector input requires Vector output
            
        """
        
        N = X.ncomp

        # Calculate curl
        # 3D input yields 3D output
        if N == 3:
            output[0]['kspace'] = X[2].deriv(self._trans[1]) - X[1].deriv(self._trans[2])
            output[1]['kspace'] = X[0].deriv(self._trans[2]) - X[2].deriv(self._trans[0])
            output[2]['kspace'] = X[1].deriv(self._trans[0]) - X[0].deriv(self._trans[1])
        
        # 2D input (xy) yields scalar output (z)
        elif N == 2:
            output['kspace'] = X[1].deriv(self._trans[0]) - X[0].deriv(self._trans[1])    
            
        # Scalar input (z) yeilds vector output (xy)
        elif N == 1:
            output[0]['kspace'] = X[0].deriv(self._trans[1])
            output[1]['kspace'] = -X[0].deriv(self._trans[0])
        
class Hydro(Physics):
    """Incompressible hydrodynamics."""
    
    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        
        # Setup data fields
        self.fields = [('u', 'VectorField')]
        self._aux_fields = [('pressure', 'VectorField'),
                            ('gradu', 'TensorField'),
                            ('ugradu', 'VectorField')]
        
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'nu': 0., 'rho0': 1.}
        self._setup_parameters(params)
        self._finalized = False

    def _finalize_init(self):
        self._setup_aux_fields(0., self._aux_fields)
        self._RHS = self.create_fields(0.)
        self._finalized = True

    def RHS(self, data):
        """
        Compute right hand side of fluid equations, populating self._RHS with
        the time derivatives of the fields.

        u_t + nu k^2 u = -ugradu - i k p / rho0

        """
        if not self._finalized:
            self._finalize_init()
        # Compute terms
        self.XgradY(data['u'], data['u'], self.aux_fields['gradu'],
                    self.aux_fields['ugradu'])
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
        self.aux_fields.time = data.time
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
            #zero_nyquist(pressure[i].data)
            pressure[i].zero_nyquist()

class ShearHydro(Hydro):
    def RHS(self, data):
        Hydro.RHS(self, data)

        self._RHS['u']['x']['kspace'] += 2.*self.parameters['Omega']*data['u']['y']['kspace']
        self._RHS['u']['y']['kspace'] += -(2*self.parameters['Omega'] + self.parameters['S'])*data['u']['x']['kspace']
        
        return self._RHS

    def pressure(self, data):
        """
        Compute pressure term for ufields: i k p / rho0
        
        p / rho0 = i (k * ugradu + rotation + shear)/ k^2
        ==> pressure term = - k (k * ugradu + rotation + shear) / k^2
        
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

        tmp += 2. * self.parameters['S'] * data['u']['x'].k['y'] * data['u']['x']['kspace'] - 2. * self.parameters['Omega'] * data['u']['y'].k['x'] * data['u']['y']['kspace'] + 2 * self.parameters['Omega'] * data['u']['x'].k['y'] * data['u']['x']['kspace']

        # Construct full term
        for i in self.dims:            
            pressure[i]['kspace'] = -data['u'][i].k[self._trans[i]] * tmp / k2
            #zero_nyquist(pressure[i].data)
            pressure[i].zero_nyquist()

class MHD(Hydro):
    """Incompressible magnetohydrodynamics."""
    
    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        
        # Setup data fields
        self.fields = [('u', 'VectorField'),
                       ('B', 'VectorField')]
        self._aux_fields = [('Ptotal', 'VectorField'),
                            ('mathtmp', 'ScalarField'),
                            ('ugradu', 'VectorField'),
                            ('BgradB', 'VectorField'),
                            ('ugradB', 'VectorField'),
                            ('Bgradu', 'VectorField')]
        
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'nu': 0., 'rho0': 1., 'eta': 0.}

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
        mathtmp = self.aux_fields['mathtmp']
        ugradu = self.aux_fields['ugradu']
        BgradB = self.aux_fields['BgradB']
        ugradB = self.aux_fields['ugradB']
        Bgradu = self.aux_fields['Bgradu']
        Ptotal = self.aux_fields['Ptotal']
        pr4 = 4 * na.pi * self.parameters['rho0']
        
        # Compute terms
        self.XlistgradY([u, B], u, mathtmp, [ugradu, Bgradu])
        self.XlistgradY([u, B], B, mathtmp, [ugradB, BgradB])
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
            #zero_nyquist(Ptotal[i].data)
            Ptotal[i].zero_nyquist()
    
            
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
    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        self.fields = [('delta', 'ScalarField'),
                       ('u', 'VectorField')]
        self._aux_fields = [('gradphi', 'VectorField')]
                            
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'Omega_r': 0.,
                  'Omega_m': 0.,
                  'Omega_l': 0.,
                  'H0': 100.}
        self._setup_parameters(params)
        self._setup_aux_fields(0., self._aux_fields)
        aux_eqn_rhs = lambda a: a*friedmann(a, self.parameters['H0'],
                                            self.parameters['Omega_r'],
                                            self.parameters['Omega_m'], 
                                            self.parameters['Omega_l'])
        
        self._setup_aux_eqns(['a'],[aux_eqn_rhs], [0.002])

        self._RHS = self.create_fields(0.)

    def RHS(self, data):
        self.density_RHS(data)
        self.vel_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def density_RHS(self, data):
        a = self.aux_eqns['a'].value
        divu = na.zeros(data['delta'].shape, complex)
        for i in self.dims:
            divu += data['u'][i].deriv(self._trans[i])

        self._RHS['delta']['kspace'] = -divu / a

    def vel_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']
        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)
        for i in self.dims:
            self._RHS['u'][i]['kspace'] = (-gradphi[i]['kspace'] - 
                                            adot * data['u'][i]['kspace']) / a
        
    def grad_phi(self, data):
        a = self.aux_eqns['a'].value
        H = self.aux_eqns['a'].RHS(a) / a

        gradphi = self.aux_fields['gradphi']
        tmp = (-3./2. * H*H * data['delta']['kspace'] /
                data['delta'].k2(no_zero=True))
        
        for i in self.dims:            
            gradphi[i]['kspace'] = 1j * a*a * data['u'][i].k[self._trans[i]] * tmp

class CollisionlessCosmology(LinearCollisionlessCosmology):
    """This class implements collisionless cosmology with nonlinear terms.

    solves
    ------
    d_t d = -(1 + d) div(v_) / a - v dot grad(d) / a
    d_t v = -grad(Phi) / a - H v -(v dot grad) v / a
    laplacian(Phi) = 3 H^2 d / 2
    """

    def __init__(self, *args, **kwargs):
        LinearCollisionlessCosmology.__init__(self, *args, **kwargs)
        self._aux_fields.append(('graddelta', 'VectorField'))
        self._aux_fields.append(('ugraddelta', 'ScalarField'))
        self._aux_fields.append(('divu', 'ScalarField'))
        self._aux_fields.append(('deltadivu', 'ScalarField'))
        self._aux_fields.append(('gradu', 'TensorField'))
        self._aux_fields.append(('ugradu', 'VectorField'))
        self._setup_aux_fields(0., self._aux_fields) # re-creates gradphi

    def div_u(self, data):
        # should really calculate gradu first, then get divu from that
        divu = self.aux_fields['divu']

        divu['kspace'] = na.zeros_like(divu['kspace'])
        for i in self.dims:
            divu['kspace'] += data['u'][i].deriv(self._trans[i])
    
    def delta_div_u(self, data, compute_divu=False):
        """calculate delta*div(v) in x-space, using 2/3 de-aliasing 

        """

        if compute_divu:
            self.div_u(data)
        divu = self.aux_fields['divu']
        deltadivu = self.aux_fields['deltadivu']
        
        deltadivu['xspace'] = divu['xspace'] * data['delta']['xspace']

    def RHS(self, data):
        self.density_RHS(data)
        self.vel_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def density_RHS(self, data):
        a = self.aux_eqns['a'].value
        self.div_u(data)
        self.XgradY(data['u'], data['delta'], self.aux_fields['graddelta'], 
               self.aux_fields['ugraddelta'])
        self.delta_div_u(data)
        divu = self.aux_fields['divu']
        ugraddelta = self.aux_fields['ugraddelta']
        deltadivu = self.aux_fields['deltadivu']

        self._RHS['delta']['kspace'] = -(divu['kspace'] + 
                                         deltadivu['kspace'] + 
                                         ugraddelta['kspace']) / a
        
    def vel_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        self.XgradY(data['u'], data['u'], self.aux_fields['gradu'], 
                    self.aux_fields['ugradu']) 
        
        ugradu = self.aux_fields['ugradu'] 
        
        for i in self.dims:
            self._RHS['u'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                            adot * data['u'][i]['kspace'] +
                                            ugradu[i]['kspace']) / a
        
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
