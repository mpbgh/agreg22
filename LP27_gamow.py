#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Résout l'équation de schrodinger avec une barrière de potentiel
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# paquet d'onde incident non normalisé
def paquetOndeIncident (x, xCentre, largeur, k):
    paquet = np.exp (- (x - xCentre)**2 / (2*largeur**2))
    propagation = np.exp (1j * k * x)
    return paquet * propagation

# barrière de potentiel "carrée"
def barriere (x, xCentre, largeur, amplitude):
    carree = np.exp (-np.power ((x - xCentre) / largeur, 128))
    return amplitude * carree

# barrière gamow
def barriereGamow (x, xCentre, amplitude):
    coulomb = (1 + np.sign (x - xCentre)) / x**2
    forte = 10 * (1 - np.sign (x - xCentre))
    return amplitude * (coulomb - forte)

# operateur d'évolution
def operateurEvolution (I, H, dt):
    return (I - 1j*dt/2. * H)

# trace la barrière
def traceBarriere (V, V0):
    VMax = np.max (V)
    plt.title ("Propagation d'un paquet d'onde gaussien dans un potentiel de Gamow")
    plt.plot (x, V / V0 / 10, label='$V(x)$')
    plt.xlabel ("$x$ [unité arbitraire]")
    plt.ylabel ("$V(x) / V_0$")


"""
Initialisation
"""
# paramètres spatiaux
n = 1000
dx = 1/n
xMax = 0.5
x = np.linspace (0.01, 2*xMax, n)

# paramètres temporels
dt = 1e-5
T = 0.004
pasDeTemps = int (T / dt)

# définition de la barrière
V0 = 1.1e4
centreV = 0.3
potentiel = barriereGamow (x, centreV, V0)
fig = plt.figure ()
traceBarriere (potentiel, V0)

# définition de l'hamiltonien
Ec = (-1 / (2 * dx**2)) * sp.diags ([1, -2, 1], [-1, 0, 1], shape=(n, n))
V = sp.diags (potentiel, 0, shape=(n, n))
H = Ec + V

# onde initiale
k = 1e3
x0 = 0.1
largeur = 0.05
onde = paquetOndeIncident (x, x0, largeur, k)
# norme de l'onde
normeOnde = np.zeros ((pasDeTemps, len (onde)))
normeOnde[0,:] = abs (onde)**2


"""
Intégration temporelle
"""
# résolution de schrodinger en demi-temps pour préserver la norme
#     psi(x, t) = U(t) psi(x, 0)
# <=> U(-t/2) psi(x, t) = U(t/2) psi(x, 0)
#  => (1 + iH dt/2) psi(x, t) = (1 - iH dt/2) psi(x, 0)   si dt << 1
I =  sp.diags ([1], 0, shape=(n, n))
for i in range (pasDeTemps):
    # sp.linalg.spsolve (A, B) résout Ax = B
    onde = sp.linalg.spsolve (
        operateurEvolution (I, H, -dt),
        operateurEvolution (I, H, dt)*onde)
    normeOnde[i, :] = abs (onde)**2


# animation de la solution
line, = plt.plot ([], [], 'darkorange', label=r"$|\psi (x, t)|^2$")
def animate (i):
    line.set_data (x, normeOnde[i, :])
    return line,
plt.legend (loc='best')
anim = animation.FuncAnimation(fig, animate, frames=pasDeTemps, interval=10, blit=True, repeat=True)
plt.show()
