#-*- coding: utf-8 -*-

"""This script solves the Poisson equation -Delta V = rho by using the finite element method with quadratic elements.
The result is compared to the analytical solution and the error is displayed along with both numerical and analytical
solution."""

import matplotlib.pyplot as plt
import numpy as np

# --- Constants --- #

n = 25                          # number of finite elements
s = 2 * n                       # helpful constant
size = 0.08                     # domain size in [m]
length = size / n               # length of one element
e0 = 8.8541878176 * 10e-12      # free space permittivity [F/m]
er = 1.0                        # relative permittivity
e = e0 * er                     # total permittivity
r0 = 10e-8                      # charge density [C/m^3]
V0 = 1.0                        # el. potential at the beginning
Vend = 0.0                      # el. potential at the end
x = np.linspace(0, size, s + 1) # discrete steps for plotting

# --- Matrix A --- #

A = np.zeros((s + 1, s + 1), dtype=float)           # creates empty matrix, float

for i in range(1, s + 1, 2):                        # fills matrix appropriately
    A[i, i] = 16
    A[i - 1, i - 1] = 14
for i in range(0, s):
    A[i, i + 1] = -8
    A[i + 1, i] = -8
for i in range(0, s, 2):
    A[i, i + 2] = 1
    A[i + 2, i] = 1
A[0, 0] = 7
A[s, s] = 7

# --- Vector b --- #

b = np.ones(2 * n + 1)                              # creates vector with ones
b[1: 2 * n : 2] = 4                                 # fills the remaining values
b[2: 2 * n: 2] += 1
b = -length ** 2 * r0 * b / e / 2

# --- Boundary Conditions --- #

b -= V0 * A[:, 0]
b -= Vend * A[:, 2 * n]
b = b[1: -1]
A = A[1: -1, 1: -1]
V = np.zeros(2 * n + 1)
V[0] = V0
V[2 * n] = Vend

# --- Solution of the matrix equation --- #

V[1: -1] = np.linalg.solve(A, b)                    # solves A V = b

# --- Solution of the analytical equation --- #

Va = (r0 * x ** 2)/(2 * e) - x * ((r0 * size)/(2 * e) + V0 / size) + V0

# --- Plot commands --- #

fig, (a0, a1) = plt.subplots(2, sharex = True, gridspec_kw = {"height_ratios": [3, 1]})
a0.plot(x, V, label = "numerical")
a0.plot(x, Va, "*",label = "analytical")
a1.plot(x, abs(V - Va), label = "error", color = "r")
a0.legend()
a1.legend()
plt.show()