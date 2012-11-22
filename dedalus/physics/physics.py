"""
Physics classes for defining fields and providing a right hand side
to time integrators.

Authors: J. S. Oishi <jsoishi@gmail.com>
         K. J. Burns <keaton.burns@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011, 2012 J. S. Oishi, K. J. Burns.  All Rights Reserved.

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


import numpy as np
from dedalus.utils.logger import mylog
from dedalus.config import decfg
from dedalus.data_objects.api import create_field_classes, AuxEquation, StateData


def _reconstruct_object(*args, **kwargs):
    new_args = [args[1]['shape'], args[1]['_representation'], args[1]['length']]
    obj = args[0](*new_args)
    obj.__dict__.update(args[1])
    return obj


class Physics(object):
    """
    Base class for a physics object. Defines fields and provides
    a right hand side for the time integration scheme.

    """

    def __init__(self, shape, representation, length=None):
        """
        Base class for a physics object. Defines fields and provides
        a right hand side for the time integration scheme.

        Parameters
        ----------
        shape : tuple of ints
            The shape of the data in xspace: (z, y, x) or (y, x)
        representation : class
            A Dedalus representation class
        length : tuple of floats, optional
            The length of the data in xspace: (z, y, x) or (y, x),
            defaults to 2 pi in all directions.

        """

        # Store inputs
        self.shape = shape
        self._representation = representation
        if length:
            self.length = length
        else:
            self.length = (2 * np.pi,) * len(self.shape)

        # Dimensionality
        self.ndim = len(self.shape)
        self.dims = xrange(self.ndim)

        # Setup containers
        self._field_list = []
        self._aux_field_list = []
        self.parameters = {}
        self.aux_eqns = {}
        self.forcing_functions = {}
        self._forcing_function_names = {}

        # Additional setup
        self._field_classes = create_field_classes(
                self._representation, self.shape, self.length)
        self._is_finalized = False
        self._tracer = decfg.getboolean('physics','use_tracer')

    def __getitem__(self, item):
         value = self.parameters.get(item, None)
         if value is None:
              raise KeyError
         return value

    def __reduce__(self):
        savedict = {}
        exclude = ['aux_fields', '_field_classes', 'forcing_functions']
        self._is_finalized = False
        for k,v in self.__dict__.iteritems():
            if k not in exclude:
                savedict[k] = v
        return (_reconstruct_object, (self.__class__, savedict))

    def _finalize(self):
        self._setup_aux_fields(0., self._aux_field_list)
        self._is_finalized = True

    def create_fields(self, time, field_list=None):
        if field_list == None:
            field_list = self._field_list

        return StateData(time, self.shape, self.length, self._field_classes,
                         field_list=field_list, params=self.parameters)

    def _setup_aux_fields(self, t, aux_field_list=None):
        self.aux_fields = self.create_fields(t, aux_field_list)

    def _setup_aux_eqns(self, aux_eqns, RHS, ics, kwarglists):
        """ create auxiliary ODEs to be solved alongside the spatial gradients.

        inputs
        ------
        aux_eqns -- a list of names for the equations
        RHS --

        """
        for f, r, ic, kwargs in zip(aux_eqns, RHS, ics, kwarglists):
            self.aux_eqns[f] = AuxEquation(r, kwargs, ic)

    def RHS(self, data, deriv):

        # Finalize
        if not self._is_finalized:
            self._finalize()

        # Zero derivative fields
        for fname, field in deriv:
            field.zero_all()

        for fname, field in self.aux_fields:
            field.zero_all(space='kspace')

        # Synchronize times
        deriv.set_time(data.time)
        self.aux_fields.set_time(data.time)

    def gradX(self, X, output):
        """
        Compute Jacobian: gradX[N * i + j] = dX_i/dx_j

        Parameters
        ----------
        X : Scalar/VectorField object
            Input field
        output : Vector/TensorField object
            Field for storing output

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

    def XgradY(self, X, Y, stmp, vtmp, output):
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

        # Zero output field
        output.zero_all('xspace')

        # Copy X
        for cindex, comp in X:
            vtmp[cindex]['kspace'] = comp['kspace']
            #vtmp[cindex].require_space('xspace')

        # Construct XgradY
        for cindex, comp in Y:
            for i in self.dims:
                stmp['kspace'] = comp.deriv(self._trans[i])
                output[cindex]['xspace'] += stmp['xspace'] * vtmp[i]['xspace']
                #np.add(output[cindex]['xspace'],
                #       stmp['xspace'] * vtmp[i]['xspace'],
                #       output[cindex]['xspace'])

    def XlistgradY(self, Xlist, Y, stmp, vtmp, outlist):
        """
        Calculate X dot (grad Y) for X in Xlist.
        This is a low-memory alternative to XgradY (never stores a full gradY tensor).

        Inputs:
            Xlist       List of input VectorField objects
            Y           Input Scalar/VectorField object
            stmp        List of ScalarField object for use in internal calculations
            vtmp        VectorField object for use in internal calculations
            outlist     List of output Scalar/VectorField object

        """

        N = self.ndim

        # Zero all output fields
        for outfield in outlist:
            outfield.zero_all('xspace')

        for i,X in enumerate(Xlist):
            for j in xrange(X.ncomp):
                vtmp[i][j]['kspace'] = X[j]['kspace']
                vtmp[i][j]['xspace']

        for i in xrange(Y.ncomp):
            for j in self.dims:
                # Compute dY_i/dx_j
                stmp['kspace'] = Y[i].deriv(self._trans[j])

                # Add term to each output
                for k,X in enumerate(vtmp):
                    #outlist[k][i]['xspace'] += (X[j]['xspace'] * stmp['xspace']).real
                    np.add(outlist[k][i]['xspace'], (X[j]['xspace'] * stmp['xspace']), outlist[k][i]['xspace'])


    def XconstcrossY(self, X, Y, output):
        """
        Calculate X cross Y, where X is a constant, computed in kspace.

        Parameters
        ----------
        X : float or array of floats
        Y : VectorField object
        output : Scalar/VectorField object

        Notes
        -----
        In 2D, if X is a float/array then output must be a Vector/ScalarField.

        """

        # X references
        if self.ndim == 2:
            if np.isscalar(X):
                Xz = X
            else:
                Xy, Xx = X
        else:
            Xz, Xy, Xx = X

        # Y references
        Yx = Y['x']['kspace']
        Yy = Y['y']['kspace']
        if self.ndim == 3:
            Yz = Y['z']['kspace']

        # Calculate cross product
        if self.ndim == 2:
            if np.isscalar(X):
                output['x']['kspace'] = - Xz * Yy
                output['y']['kspace'] = Xz * Yx
            else:
                output['kspace'] = Xx * Yy - Xy * Yx
        else:
            output['x']['kspace'] = Xy * Yz - Xz * Yy
            output['y']['kspace'] = Xz * Yx - Xx * Yz
            output['z']['kspace'] = Xx * Yy - Xy * Yx

    def XcrossY(self, X, Y, output):
        """
        Calculate X cross Y, computed in xspace.

        Parameters
        ----------
        X : Scalar/VectorField object
        Y : VectorField object
        output : Vector/ScalarField object

        Notes
        -----
        In 2D, if X is a ScalarField then output must be a VectorField, and
        vice versa.  In 3D, all must be VectorFields.

        """

        # X references
        if self.ndim == 2:
            if X.ncomp == 1:
                Xz = X['xspace']
            else:
                Xx = X['x']['xspace']
                Xy = X['y']['xspace']
        else:
            Xx = X['x']['xspace']
            Xy = X['y']['xspace']
            Xz = X['z']['xspace']

        # Y references
        Yx = Y['x']['xspace']
        Yy = Y['y']['xspace']
        if self.ndim == 3:
            Yz = Y['z']['xspace']

        # Calculate cross product
        if self.ndim == 2:
            if X.ncomp == 1:
                output['x']['xspace'] = - Xz * Yy
                output['y']['xspace'] = Xz * Yx
            else:
                output['xspace'] = Xx * Yy - Xy * Yx
        else:
            output['x']['xspace'] = Xy * Yz - Xz * Yy
            output['y']['xspace'] = Xz * Yx - Xx * Yz
            output['z']['xspace'] = Xx * Yy - Xy * Yx

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

        # Scalar input (z) yields vector output (xy)
        elif N == 1:
            output[0]['kspace'] = X[0].deriv(self._trans[1])
            output[1]['kspace'] = -X[0].deriv(self._trans[0])

    def laplace_solve(self, X, output):
        """
        Solve laplace(output) = X.
        """
        k2 = output.k2(no_zero=True)
        output['kspace'] = -X['kspace'] / k2


class IncompressibleHydro(Physics):
    """Homogeneous incompressible hydrodynamics."""

    def __init__(self, *args, **kwargs):
        """
        Create a physics object for incompressible hydrodynamics.

        Parameters
        ----------
        *** Set in self.parameters dictionary after instantiation. ***

        'viscosity_order' : int
            Hyperviscosity order. Defaults to 1.
        'nu' : float
            Kinematic viscosity. Defaults to 0.
        'shear_rate' : float
            Linear shearing rate S, such that v_x = S y. Defaults to 0.
        'Omega' : float, array of floats, or None
            Angular velocity vector np.array[(Omega_z, Omega_y, Omega_x)].
            Float for z-direction in 2D.  Defaults to None.

        Notes
        -----
        For a Keplerian angular velocity profile, S = 1.5 * Omega_z.

        Example
        -------
        >>> physics = IncompressibleHydro((128, 128), FourierRepresentation)
        >>> physics.parameters['viscosity_order'] = 2

        """

        # Inherited initialization
        Physics.__init__(self, *args, **kwargs)

        # Add velocity field
        self._field_list.append(('u', 'VectorField'))

        # Add auxiliary math fields
        self._aux_field_list.append(('mathscalar', 'ScalarField'))
        self._aux_field_list.append(('mathvector', 'VectorField'))

        # Add default parameters
        self.parameters['viscosity_order'] = 1
        self.parameters['nu'] = 0.
        self.parameters['shear_rate'] = 0.
        self.parameters['Omega'] = None

        # Add tracer field and parameters
        if self._tracer:
            self._field_list.append(('c', 'ScalarField'))
            self.parameters['c_diff'] = 0.


        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        self._first_rhs = True

    def __reduce__(self):

        self._first_rhs = True
        return Physics.__reduce__(self)

    def _finalize(self):

        # Inherited finalization
        Physics._finalize(self)

        # Set shear flag and check representation
        if self.parameters['shear_rate'] == 0.:
            self._shear = False
            if not self._representation._static_k:
                mylog.warning("Performance suffers when using a shearing representation without a linear shear.")
        else:
            self._shear = True
            if self._representation._static_k:
                raise ValueError("A shearing representation must be used if shear_rate is nonzero.")

        # Set rotation flag
        if self.parameters['Omega'] == None:
            self._rotation = False
        else:
            self._rotation = True
            if self.ndim == 2:
                mylog.warning("Rotation is dynamically insignificant in 2D incrompressible hydrodynamics.  Remove for optimal performance.")

    def _setup_integrating_factors(self, deriv):

        # Kinematic viscosity for u
        for cindex, comp in deriv['u']:
            nu = self.parameters['nu']
            vo = self.parameters['viscosity_order']
            if nu == 0.:
                comp.integrating_factor = None
            else:
                comp.integrating_factor = nu * comp.k2() ** vo

        # Diffusion for tracer
        if self._tracer:
            comp = deriv['c'][0]
            diff = self.parameters['c_diff']
            if diff == 0.:
                comp.integrating_factor = None
            else:
                comp.integrating_factor = diff * comp.k2() ** vo

    def RHS(self, data, deriv):
        """
        Compute right-hand side of fluid equations, cast in the form
            d_t f + S y d_x f - c div.grad(f) = RHS(f).

        RHS(u) = - u.grad(u) - S u_y e_x - grad(p) / rho0 - 2 Omega * u
        RHS(c) = - u.grad(c)

        """

        # Initial integrating factors
        if self._first_rhs:
            self._setup_integrating_factors(deriv)
            self._first_rhs = False

        # Inherited RHS
        Physics.RHS(self, data, deriv)

        # Auxiliary field references
        mathscalar = self.aux_fields['mathscalar']
        mathvector = self.aux_fields['mathvector']

        # Parameter references
        S = self.parameters['shear_rate']
        Omega = self.parameters['Omega']

        # Velocity RHS
        # Inertial term
        self.XgradY(data['u'], data['u'], mathscalar, mathvector, deriv['u'])
        for i in self.dims:
            deriv['u'][i]['kspace'] *= -1.

        # Shear term
        if self._shear:
            deriv['u']['x']['kspace'] -= S * data['u']['y']['kspace']

        # Rotation terms
        if self._rotation:
            self.XconstcrossY(Omega, data['u'], mathvector)
            for i in self.dims:
                deriv['u'][i]['kspace'] -= 2 * mathvector[i]['kspace']

        # Pressure term
        if self.__class__ == IncompressibleHydro:
            self.pressure_projection(data, deriv)

        # Tracer RHS
        # Inertial term
        if self._tracer:
            self.XgradY(data['u'], data['c'], mathscalar, mathvector, deriv['c'])
            deriv['c']['kspace'] *= -1

        # Recalculate integrating factors
        if self._shear:
            self._setup_integrating_factors(deriv)

    def pressure_projection(self, data, deriv):

        # Place references
        mathscalar = self.aux_fields['mathscalar']
        S = self.parameters['shear_rate']

        # Perform solenoidal projection
        self.divX(deriv['u'], mathscalar)
        if self._shear:
            mathscalar['kspace'] -= S * data['u']['y'].deriv('x')
        self.laplace_solve(mathscalar, mathscalar)
        for i in self.dims:
            deriv['u'][i]['kspace'] -= mathscalar.deriv(self._trans[i])


class BoussinesqHydro(IncompressibleHydro):
    """Boussinesq hydrodynamics."""

    def __init__(self, *args, **kwargs):
        """
        Create a physics object for Boussinesq hydrodynamics.

        Parameters
        ----------
        *** Set in self.parameters dictionary after instantiation. ***
        *** See IncompressibleHydro for definitions of inherited parameters. ***

        'kappa' : float
            Defaults to 0.
        'g' : float
            Defaults to 1.
        'alpha_t' : float
            Defaults to 1.
        'beta' : float
            Defaults to 1.

        """

        # Inherited initialization
        IncompressibleHydro.__init__(self, *args, **kwargs)

        # Add temperature field
        self._field_list.append(('T', 'ScalarField'))

        # Add default parameters
        self.parameters['kappa'] = 0.
        self.parameters['g'] = 1.
        self.parameters['alpha_t'] = 1.
        self.parameters['beta'] = 1.

    def _setup_integrating_factors(self, deriv):

        # Inherited integrating factors
        IncompressibleHydro._setup_integrating_factors(self, deriv)

        # Thermal diffusivity for T
        kappa = self.parameters['kappa']
        vo = self.parameters['viscosity_order']
        if kappa == 0.:
            deriv['T'][0].integrating_factor = None
        else:
            deriv['T'][0].integrating_factor = kappa * comp.k2() ** vo

    def set_thermal_forcing(self, func):
        self.forcing_functions['ThermalForcing'] = func

    def RHS(self, data, deriv):
        """
        Compute right-hand side of fluid equations, cast in the form
            d_t f + S y d_x f - c div.grad(f) = RHS(f).

        RHS(u) += g alpha_t T
        RHS(T) = - u.grad(T) - beta * u_z

        """

        # Inherited RHS
        IncompressibleHydro.RHS(self, data, deriv)

        # Auxiliary field references
        mathscalar = self.aux_fields['mathscalar']
        mathvector = self.aux_fields['mathvector']

        # Parameter references
        S = self.parameters['shear_rate']
        Omega = self.parameters['Omega']
        g = self.parameters['g']
        alpha_t = self.parameters['alpha_t']
        beta = self.parameters['beta']

        # Velocity RHS
        # Bouyancy term
        deriv['u']['z']['kspace'] += g * alpha_t * data['T']['kspace']

        # Pressure term
        if self.__class__ == BoussinesqHydro:
            self.pressure_projection(data, deriv)

        # Temperature RHS
        # Inertial term
        self.XgradY(data['u'], data['T'], mathscalar, mathvector, deriv['T'])
        deriv['T']['kspace'] *= -1.

        # Stratification term
        deriv['T']['kspace'] -= beta * data['u']['z']['kspace']

        # Thermal driving term
        if self.forcing_functions.has_key('ThermalForcing'):
            deriv['T']['kspace'] += self.forcing_functions['ThermalForcing'](data)
        deriv['T']['kspace'][0,0,0] = 0. # Must ensure (0,0,0) T mode does not grow. #REVIEW


class IncompressibleMHD(IncompressibleHydro):
    """Homogeneous incompressible magnetohydrodynamics."""

    def __init__(self, *args, **kwargs):
        """
        Create a physics object for incompressible magnetohydrodynamics.

        Parameters
        ----------
        *** Set in self.parameters dictionary after instantiation. ***
        *** See IncompressibleHydro for definitions of inherited parameters. ***

        'rho0' : float
            Background density. Defaults to 1.
        'eta' : float
            Magnetic resistivity. Defaults to 0.

        """

        # Inherited initialization
        IncompressibleHydro.__init__(self, *args, **kwargs)

        # Add magnetic field
        self._field_list.append(('B', 'VectorField'))

        # Add extra auxiliary math field
        self._aux_field_list.append(('mathvector2', 'VectorField'))

        # Add default parameters
        self.parameters['rho0'] = 1.
        self.parameters['eta'] = 0.

    def _setup_integrating_factors(self, deriv):

        # Inherited integrating factors
        IncompressibleHydro._setup_integrating_factors(self, deriv)

        # Magnetic diffusivity for B
        eta = self.parameters['eta']
        vo = self.parameters['viscosity_order']
        for cindex, comp in deriv['B']:
            if eta == 0.:
                comp.integrating_factor = None
            else:
                comp.integrating_factor = eta * comp.k2() ** vo

    def RHS(self, data, deriv):
        """
        Compute right-hand side of fluid equations, cast in the form
            d_t f + S y d_x f - c div.grad(f) = RHS(f).

        RHS(u) += curl(B) * B / (4 pi rho0)
        RHS(B) = S B_y e_x + curl(u * B)

        """

        # Inherited RHS
        IncompressibleHydro.RHS(self, data, deriv)

        # Auxiliary field references
        mathscalar = self.aux_fields['mathscalar']
        mathvector = self.aux_fields['mathvector']
        mathvector2 = self.aux_fields['mathvector2']

        # Parameter references
        S = self.parameters['shear_rate']
        rho0 = self.parameters['rho0']
        fpr = 4 * np.pi * rho0

        # Velocity RHS
        # Lorentz force
        if self.ndim == 2:
            self.curlX(data['B'], mathscalar)
            self.XcrossY(mathscalar, data['B'], mathvector)
        else:
            self.curlX(data['B'], mathvector)
            self.XcrossY(mathvector, data['B'], mathvector2)
        for i in self.dims:
            deriv['u'][i]['kspace'] += mathvector2[i]['kspace'] / fpr

        # Pressure term
        if self.__class__ == IncompressibleMHD:
            self.pressure_projection(data, deriv)

        # Magnetic field RHS
        # Induction term
        if self.ndim == 2:
            self.XcrossY(data['u'], data['B'], mathscalar)
            self.curlX(mathscalar, deriv['B'])
        else:
            self.XcrossY(data['u'], data['B'], mathvector)
            self.curlX(mathvector, deriv['B'])

        # Shear term
        if self._shear:
            deriv['B']['x']['kspace'] += S * data['B']['y']['kspace']

