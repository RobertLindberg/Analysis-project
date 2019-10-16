# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:02:07 2019

@author: m1xam
"""
from scipy import*
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import*
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from sympy.solvers import solve
from sympy import Symbol
from mpl_toolkits.mplot3d import Axes3D

'Since the df/dz is non-zero at the origin the exists a function z(x,y) defined in a neighbourhood around (0,0).'
'Solving for z in the domain [-p,p] divided into n points in both x- and y-direction, saving the solutions'
'in a 2d-array so we can plot z(x,y)'

C = []
p = 5
n = 100
for x in np.linspace(-p, p, n):
    Ctemp = []
    for y in np.linspace(-p, p, n):
        
        f = lambda z: x + 2*y + z + exp(2*z) - 1
        z0 = optimize.fsolve(f, 0, xtol = 1e-10)[0]

        Ctemp.append(z0)
    C.append(Ctemp)

A = array(C)

'Calculating partial derivatives of z by using the definition of the derivative. Using h=10^-8 for first order'
'terms, and h=10^-4 for second order terms'

f = lambda z: 1e-8+ z + exp(2*z) -1
z0 = optimize.fsolve(f, 0, xtol = 1e-10)

f = lambda z: 2e-8+ z + exp(2*z) -1
z1 = optimize.fsolve(f, 0, xtol = 1e-10)

print('dz/dx:')
print((z0-0)/(1e-8))
print('dz/dy:')
print((z1-0)/(1e-8))

f = lambda z: 2e-4 + 2e-8+ z + exp(2*z) -1
z2 = optimize.fsolve(f, 0, xtol = 1e-10)

f = lambda z: 2e-4 + z + exp(2*z) -1
z3 = optimize.fsolve(f, 0, xtol = 1e-10)

f = lambda z: 1e-4 + 1e-8+ z + exp(2*z) -1
z4 = optimize.fsolve(f, 0, xtol = 1e-10)

f = lambda z: 1e-4 + z + exp(2*z) -1
z5 = optimize.fsolve(f, 0, xtol = 1e-10)

print('d^2z/dx^2')
print((((z4-z5)-(z0-0))/(1e-8))/(1e-4))
print('d^2z/dy^2')
print((((z2-z3)-(z1-0))/(1e-8))/(1e-4))

'All partial derivates of order 2 and 1 has been calculated except d^2z/dxdy. We solve it'
'by rewriting z=(Ax+By+Cx^2+Dxy+Ey^2), and taylor expanding x + 2*y + z + exp(2*z) - 1 = 0.'
'Inserting our numerically derived constants A,B,C,E we can easily solve for D by hand. We get'
'D=E'

'Defining the 2nd order taylor polynomial T(x,y) of z(x,y) and saving it as a 2d-array:'

def T(x,y):
    return -x/3-(2*y)/3-(0.59266424*y**2)/2-(0.14814578/2)*x**2-(0.59266424/2)*x*y

x = np.linspace(-p,p,n) 
y = np.linspace(-p,p,n)

X, Y = np.meshgrid(x, y)
Z = T(X, Y)

'Calculating the relative error of the taylorapproximation: |(z(x,y)-T(x,y))/z(x,y)|.'
'Since it has z(x,y) in the denominator the graph will have large spikes whenever z(x,y)'
'is close to 0 and T(x,y) is not. This makes it impossible to get a good grasp of the'
'error by just looking at the plot. Hence we introduce a tolerance term, an upper bound'
'for the allowed relative error to be plotted.'

tol = 10
T1 = []
T2 = []
for i in range(n):
    Ttemp1 = []
    Ttemp2 = []
    for j in range(n):
        Ttemp1.append((abs((A[i][j]-Z[i][j])/A[i][j])))
        if (abs((A[i][j]-Z[i][j])/A[i][j]))<tol:
            
            Ttemp2.append((abs((A[i][j]-Z[i][j])/A[i][j])))
        else:
            Ttemp2.append(tol)
    T1.append(Ttemp1)
    T2.append(Ttemp2)
    
T = array(T1)
Ttol = array(T2)

'Plotting our results'

fig = plt.figure(1)

ax = plt.axes(projection = '3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('z(x,y)')

ax.plot_surface(X,Y,A, cmap='jet', rstride = 1, cstride = 1)

fig = plt.figure(2)

ax = plt.axes(projection = '3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Taylorapproximation of z(x,y)')

ax.plot_surface(X,Y,Z, cmap='jet', rstride = 1,cstride = 1)

fig = plt.figure(3)

ax = plt.axes(projection = '3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Relative error without tolerance')

ax.plot_surface(X,Y,T, cmap='jet', rstride = 1,cstride = 1)

fig = plt.figure(4)

ax = plt.axes(projection = '3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Relative error with tolerance')

ax.plot_surface(X,Y,Ttol, cmap='jet', rstride = 1,cstride = 1)