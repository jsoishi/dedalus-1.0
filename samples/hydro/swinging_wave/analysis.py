"""
Plot swinging wave amplitude vs. time and compare to analytic results of 
Lithwick (2007).

Author: K. J. Burns <keaton.burns@gmail.com>
Affiliation: UC Berkeley
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

import numpy as np
import matplotlib.pyplot as plt

# Setup wave parameters
kx, ky = (0.01, 4)
w = 0.01
Omega = 1.
S = -1.5
q = -S * Omega

# Simulated mode amplitudes
data_ux = np.loadtxt('ux_mode_amplitudes.dat', skiprows=2, delimiter='\t', dtype=np.complex128)
data_uy = np.loadtxt('uy_mode_amplitudes.dat', skiprows=2, delimiter='\t', dtype=np.complex128)
s_t = data_ux[:, 0].real
s_ux = 2 * data_ux[:, 1]
s_uy = 2 * data_uy[:, 1]

# Analytic mode amplitudes, Lithqick Eq. (23)
func_ux = lambda t: 1j * w * (ky - q * kx * t) / (kx ** 2 + (ky - q * kx * t) ** 2)
func_uy = lambda t: -1j * w * kx / (kx ** 2 + (ky - q * kx * t) ** 2)
a_t = np.linspace(s_t.min(), s_t.max(), s_t.size * 10)
a_ux = func_ux(a_t)
a_uy = func_uy(a_t)

# Plot
fig, axes = plt.subplots(2, 1)

axes[0].plot(a_t, a_ux.imag, '-k', label='Lithwick')
axes[0].plot(s_t, s_ux.imag, '.b', label='Dedalus')
axes[0].legend()
axes[0].set_ylabel(r'$\Im(\hat{u}_x)$')

axes[1].semilogy(a_t, np.abs(a_uy.imag), '-k')
axes[1].semilogy(s_t, np.abs(s_uy.imag), '.b')
axes[1].set_xlabel(r'$t$')
axes[1].set_ylabel(r'$|\Im(\hat{u}_y)|$') 

fig.savefig("mode_amplification.png")
