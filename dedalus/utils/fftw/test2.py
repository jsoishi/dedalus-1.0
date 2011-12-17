import time
import numpy as np
import pylab as P
from fftw import PlanAxes

size = 128
data = np.zeros((size,size,size), dtype='complex')

p = PlanAxes(data)


