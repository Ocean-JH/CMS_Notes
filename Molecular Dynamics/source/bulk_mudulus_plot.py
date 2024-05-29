#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv('fcc_bulk_modulus.csv')
v = np.array(df['V'])
e = np.array(df['E'])

a, b, c = np.polyfit(v, e, 2)
v0 = -b / (2 * a)
e0 = a * v0 ** 2 + b * v0 + c
b0 = 2 * a * v0
bP = 3.5
x0 = [e0, b0, bP, v0]


# Birch–Murnaghan equation of state
def Murnaghan(parameters, vol):
    E0 = parameters[0]
    B0 = parameters[1]
    BP = parameters[2]
    V0 = parameters[3]
    E = E0 + B0 * vol / BP * (((V0 / vol) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1.)

    return E


# error
def residual(pars, y, x):
    err = y - Murnaghan(pars, x)

    return err


from scipy.optimize import leastsq

murnpars, ier = leastsq(residual, x0, args=(e, v))

print('Bulk Modulus:' + str(murnpars[1]))
print('lattice constant:', murnpars[3] ** (1 / 3))

from matplotlib import pyplot as plt

v_mesh = np.linspace(np.min(v), np.max(v), 1000)
plt.scatter(v, e, 10)
plt.plot(v_mesh, Murnaghan(murnpars, v_mesh))
plt.ylabel('Energy')
plt.xlabel('Volume')
plt.show()
