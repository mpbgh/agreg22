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
    return np.exp (-np.power ((x - xCentre) / largeur, 128)) * amplitude

# operateur d'évolution
def operateurEvolution (I, H, dt):
    return (I - 1j*dt/2. * H)

# trace la barrière
def traceBarriere (V, V0):
    plt.title ("Propagation d'un paquet d'onde gaussien")
    plt.plot (x, potentiel / V0, label='$V(x)$')
    plt.xlabel ("$x$ [unité arbitraire]")
    plt.ylabel ("$V(x) / V_0$")
    #plt.grid ()

"""
Initialisation
"""
# paramètres spatiaux
n = 1000
dx = 1/n
xMax = 0.5
x = np.linspace (-xMax, xMax, n)

# paramètres temporels
dt = 1e-5
T = 0.005
pasDeTemps = int (T / dt)

# définition de la barrière
V0 = 4.55e5
centreV = 0.0
largeurV = 0.02
potentiel = barriere (x, centreV, largeurV, V0)
fig = plt.figure ()
traceBarriere (potentiel, V0)

# définition de l'hamiltonien
Ec = (-1 / (2 * dx**2)) * sp.diags ([1, -2, 1], [-1, 0, 1], shape=(n, n))
V = sp.diags (potentiel, 0, shape=(n, n))
H = Ec + V

# onde initiale
k = 1e3
x0 = -0.2
largeur = 0.05
onde = paquetOndeIncident (x, x0, largeur, k)
# norme de l'onde
normeOnde = np.zeros ((pasDeTemps, len (onde)))
normeOnde[0,:] = abs (onde)**2
plt.ylim (-0.1, 2)


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
