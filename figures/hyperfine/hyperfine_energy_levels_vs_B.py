# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:26:54 2022

@author: kolkowitz
"""

import numpy
import matplotlib.pyplot as plt

D = 2870
P = -4.95
Azz = 2.2
Ye = 28025 / 1e4
Yn = 3.077 / 1e4


def ham(B, S, I):
    return D * S**2 + Ye * B * S + Yn * B * I + P * I**2 + S * Azz * I


lin_b = numpy.linspace(0, 100, 100)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

sz_iz = ham(lin_b, 0, 0)
sz_ip = ham(lin_b, 0, 1)
sz_im = ham(lin_b, 0, -1)

sp_iz = ham(lin_b, 1, 0)
sp_ip = ham(lin_b, 1, 1)
sp_im = ham(lin_b, 1, -1)

sm_iz = ham(lin_b, -1, 0)
sm_ip = ham(lin_b, -1, 1)
sm_im = ham(lin_b, -1, -1)

p_p = sp_ip - sz_ip
p_z = sp_iz - sz_iz
p_m = sp_im - sz_im

m_p = sm_ip - sz_ip
m_z = sm_iz - sz_iz
m_m = sm_im - sz_im

ax.plot(lin_b, ham(lin_b, 0, 0), "r", label=r"$m_s = 0, m_I = 0$")
ax.plot(lin_b, ham(lin_b, 0, 1), "r--", label=r"$m_s = 0, m_I = +1$")
ax.plot(lin_b, ham(lin_b, 0, -1), "r-.", label=r"$m_s = 0, m_I = -1$")

ax.plot(lin_b, ham(lin_b, 1, 0), "b", label=r"$m_s = +1, m_I = 0$")
ax.plot(lin_b, ham(lin_b, 1, 1), "b--", label=r"$m_s = +1, m_I = +1$")
ax.plot(lin_b, ham(lin_b, 1, -1), "b-.", label=r"$m_s = +1, m_I = -1$")

ax.plot(lin_b, ham(lin_b, -1, 0), "g", label=r"$m_s = -1, m_I = 0$")
ax.plot(lin_b, ham(lin_b, -1, 1), "g--", label=r"$m_s = -1, m_I = +1$")
ax.plot(lin_b, ham(lin_b, -1, -1), "g-.", label=r"$m_s = -1, m_I = -1$")

# ax.plot(lin_b, p_z,'b',label=r'$m_s = +1, m_I = 0$')
# ax.plot(lin_b, p_p,'b--',label=r'$m_s = +1, m_I = +1$')
# ax.plot(lin_b, p_m,'b-.',label=r'$m_s = +1, m_I = -1$')

# ax.plot(lin_b, m_z,'g',label=r'$m_s = -1, m_I = 0$')
# ax.plot(lin_b, m_p,'g--',label=r'$m_s = -1, m_I = +1$')
# ax.plot(lin_b, m_m,'g-.',label=r'$m_s = -1, m_I = -1$')

ax.set_xlabel(r"Magnetic field (G)")
ax.set_ylabel("Contrast (arb. units)")
ax.legend()
plt.show()
