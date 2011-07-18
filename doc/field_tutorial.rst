===================================
A brief Primer on fields in Dedalus
===================================

Assume in the following that 

data = StateData(...)

Vector/Tensors
--------------
Then accessing

data['u'] will give a {HydroVectorField,
LinearCollisionlesCosmologyVectorField} or whatever is dyamically
created when the Physics class is instantiated.

data['u']['x'] --> FourierRepresentation

data['u']['x'].k2() --> square wavenumbers
data['u']['x'].deriv('y') --> d u_x/ dy

a = data['u']['x']['kspace'] --> numpy array with data
data['u']['x']['kspace'] = [0,1l2,3,3] ---> __setitem__ on FourierRepresnetation

data['u']['x']['xspace']
data['u']['x']['xspace'] = data['u']['x']['xspace'] * [0,1l2,3,3] ---> NB: DOES NOT TAKE FOURIER TRANSFORM

Scalars
-------

data['density'] --> will LOOK like a FourierRepresentation but IS a HydroScalarField

HydroScalarField.__getitem__  takes a "space" argument:

data['density']['kspace']


without the __getitem__, the ScalarField will search the attributes of its underlying representation:

data['density'].k2() --> this will work
data['density'].deriv('x') --> this will also work
