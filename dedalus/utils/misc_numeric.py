""" Simple tools for integration, interpolation, and root-finding, to remove SciPy dependency.

"""
import numpy as na

def integrate_quad(f,x):
    """integrates f with quadratic interpolation over sample points x.
    Relative error of ~2e-6 compared with SciPy's integrate.simps on 
    typical cosmology normalization data.

    """
    integral = 0
    for i in range((len(x)-1)/2):
        x1 = x[2*i]
        x2 = x[2*i+1]
        x3 = x[2*i+2]
        y1 = f[2*i]
        y2 = f[2*i+1]
        y3 = f[2*i+2]
        a = ((y2-y1)*(x1-x3) + (y3-y1)*(x2-x1))/((x1-x3)*(x2**2-x1**2) + 
                                                 (x2-x1)*(x3**2-x1**2))
        b = ((y2-y1) - a*(x2**2-x1**2))/(x2-x1)
        c = y1 - a*x1**2 - b*x1
        integral += a/3*(x3**3 - x1**3) + b/2*(x3**2 - x1**2) + c*(x3 - x1)
    # If we have an even number of points, use trapezoid for the last interval
    if (len(x) % 2) == 0:
        integral += (x[len(x)-1]-x[len(x)-2])*(f[len(x)-1]+f[len(x)-2])/2
    return integral

def integrate_simp(f,x_0, x_f, n):
    """integrates f from x_0 to x_f with n intervals using Simpson's rule. Error should go as n^-5."""
    integral = 0
    x = [x_0 + i*1./n*(x_f - x_0) for i in range(int(n)+1)]
    for i in range((len(x)-1)/2):
        integral += ((x[2*i+2]-x[2*i])/6 * 
                     (f(x[2*i]) + 4*f(x[2*i+1]) + f(x[2*i+2])))
    # If we have an even number of points, use trapezoid for the last interval
    if (len(x) % 2) == 0:
        integral += (x[len(x)-1]-x[len(x)-2])*(f(len(x)-1)+f(len(x)-2))/2
    return integral

def interp_linear(x, f):
    """return function for linear interpolation of f sampled at points x 
    (where x is strictly increasing). The returned function returns zero 
    outside interpolation range.

    """
    piecewise = [lambda z: 0,]*(len(x)-1)
    xleft = x[0:len(x)-1]
    xright = x[1:]
    piecewise = lambda i,z: f[i] + (z - x[i])*(f[i+1]-f[i])/(x[i+1]-x[i])
    # function to return index of interval containing z
    in_interval = lambda z: na.nonzero((xleft <= z)&(z < xright))[0]
    # function to evaluate f at array of points
    f_lin = lambda zz: na.reshape([piecewise(in_interval(z),z) 
                                  for z in zz.flatten(1)], zz.shape)
    return f_lin

def find_zero(func, left, right, eps_abs = 1e-10):
    """find the zero of a function using the bisection method
    (binary search). Assumes a single zero.

    Input:
        func          function to find zeros for
        left          left endpoint(s) of range to search. If func 
                      is a function of nvar variables then left should
                      be an array of length nvar.
        right         right endpoint(s) of range to search
        eps_abs       bound on absolute error on solution

    Output:
        y             solution satisfying f(y) = 0

    """
    nvar = len(left)
    err = max(right - left)
    while err > eps_abs:
        midpoint = (left + right) / 2.
        f_left = func(left)
        f_right = func(right)
        f_midpoint = func(midpoint)
        
        for i in range(nvar):
            if f_midpoint[i] == 0:
                left[i] = midpoint[i]
                right[i] = midpoint[i]
            elif na.sign(f_midpoint[i]) == na.sign(f_left[i]):
                left[i] = midpoint[i]
            else: 
                right[i] = midpoint[i]
        err = max(right - left)
    return midpoint

