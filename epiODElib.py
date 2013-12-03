###############################################################################
###############################################################################
#   Copyright 2013 Kyle S. Hickmann and
#                  The Administrators of the Tulane Educational Fund
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
###############################################################################
###############################################################################

###############################################################################
###############################################################################
# This library contains a suite of functions to generate simulations 
# of stochastic and deterministic disease progressions with and without 
# noise due to model error. 
###############################################################################
###############################################################################
import numpy as np 
import scipy.linalg as linalg
import numpy.random as rn
import math 

###############################################################################
# Use Euler-Maruyama method on SIR SDE with noise.
# SDE: dX = A(X)*dt + B(X)*dW1 + sigma*dW2, X(0) = X0
# X = (S,I)^T
# A(X) = [-(beta)*S*I; (beta)*S*I-mu*I];
# B(X) = sqrtm([(beta)*S*I, -(beta)*S*I; -(beta)*S*I, (beta)*S*I+mu*I];
# dW1,2 = (dq,dp)^T = Independent, uncorrelated Brownian increment
#      vector.  
#
# Given: 
# - initial conditions, transmission rates, recovery rates, total population
# - time vector of deterministic steps
# - number of stochastic steps per deterministic step
# - model error standard deviation
#
# Generate: 
# - Ensemble of N SIR simulations of length equal to number 
#   of stochastic time steps.
###############################################################################
def SIRsde_noise(y0, time, Nstep, beta, mu, sigma):
    # y0 = 1x2 numpy array of (S0,I0)^T initial condition
    # time = time numpy vector containing the deterministic time points
    # Nstep = Number of stochastic time steps per deterministic step
    # beta = scalar transmission rate
    # mu = scalar recovery rate
    # sigma = model error standard deviation
    
    Tsteps = time.shape[0]
    # Xsim = size 2x(Tsteps) numpy array of simulation
    Xsim = np.zeros((2,Tsteps))
    Xsim[0,0] = y0[0].copy() 
    Xsim[1,0] = y0[1].copy() 

    # Define time steps
    # Deterministic time steps
    Dt = time[1]-time[0]
    dt = Dt/float(Nstep)
    
    # Create temporary storage vector for the iteration
    Xtmp = np.zeros((2))
    for i in range(int(Tsteps-1)):
        Xtmp = Xsim[:,i].copy()
        Stmp = Xsim[0,i].copy()
        Itmp = Xsim[1,i].copy()
        
        # Create if statement to maintain non-negativity of Infected number
        if (Itmp <= 0):
            Xsim[0,i+1] = Stmp
            Xsim[1,i+1] = 0.
        else:
            # Create Brownian increments
            dW1 = math.sqrt(dt)*rn.randn(2,Nstep)
            dW2 = math.sqrt(dt)*rn.randn(2,Nstep)

            # Create Brownian integrated step
            Winc1 = dW1.sum(1)
            Winc2 = dW2.sum(1)
        
            # Create vector and array of SIR equation
            A = np.array([-(beta)*Stmp*Itmp, (beta)*Stmp*Itmp-mu*Itmp])

            B = np.array([[(beta)*Stmp*Itmp, -(beta)*Stmp*Itmp],
                         [-(beta)*Stmp*Itmp, (beta)*Stmp*Itmp+mu*Itmp]])

            # Matrix square root using Cholesky decomposition
            sqrtB = linalg.cholesky(B,lower=1)
  
            Xsim[:,i+1] = Xtmp + Dt*A + np.dot(sqrtB,Winc1) + sigma*Winc2 
    
    return Xsim

###############################################################################
# Simulate a very basic deterministic SIR system with parameters
# beta = transmission rate
# mu = recovery rate
# Returns numpy array:
#    Xsim = (2)x(Ntimestep) first column S(t), second column I(t).
def SIRode(y0, time, beta, mu): 
    # y0 = 2x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # mu = scalar
    # Npop = scalar
    
    Xsim = rk4(SIR_D, y0, time, args=(beta,mu,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for basic SIR system.
def SIR_D(y,t,beta,mu):
    # y[0] = susceptible
    # y[1] = infected
    dy = np.zeros((2,1))
    
    dy[0] = -(beta)*y[0]*y[1]
    dy[1] = (beta)*y[0]*y[1] - mu*y[1]
    return dy

###############################################################################
# Used in several of the above simulations.
#
# Integrate an ODE specified by 
# y' = f(y,t,args), y(t0) = y0, t0 < t < T
# Using fourth order Runge-Kutta on a specified time grid.

# f() must return a numpy column array of outputs of the same
# dimension as that of y. It must take a numpy column array of
# variable values y, a single float time t, and any other arguments.

# The output is a numpy array of y values at the times in t of
# dimensions (len(t),len(y0)).
def rk4(func, y0, t, args=()):
    output = np.zeros((len(t),len(y0)))
    output[0,:] = y0[:,0].copy()
    yiter = y0.copy()

    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        k1 = h*func(yiter, t[i], *args)
        k2 = h*func(yiter + .5*k1, t[i] + .5*h, *args)
        k3 = h*func(yiter + .5*k2, t[i] + .5*h, *args)
        k4 = h*func(yiter + k3, t[i] + h, *args)
        yiter = yiter + 0.16666666*(k1 + 2*k2 + 2*k3 + k4)
        output[i+1,:] = yiter[:,0].copy()

    return output
