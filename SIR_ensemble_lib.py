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

# This library has functions for SIR ensemble generation,
# ensemble-to-measurement mapping, and Ensemble reshaping. It is meant
# to be used in conjunction with the kflib.py or pflib.py

# Functions:
#    SIRens = Function to generate the SIR ensemble from parameter sets. 
#             The ensemble returned has correct form to pass to an analysis.
#             Also used to restore SIR balances in analysis after enKF step.
#
#    SIRmeasure = Function to map ensemble to measurements. Measurements 
#                 returned is in form to be passed to an analysis.
#    
#    SIRshape = Function to reformat Ensemble array and time vector to a 
#               form easy for AssimViz.py to graph.

import numpy as np 
import scipy.linalg as linalg
import numpy.random as rn
import math 

# User defined package
import epiODElib
###################################################################
#                      HELPER FUNCTIONS                           #
###################################################################
# Functions to generate ensemble given parameter file and time span.
# Also used to restore SIR balance of analysis ensemble after KF step.
#
# RETURNS:
#    EnsArray = (2*Ntimesteps)x(EnSize) numpy array. Column is 
#               (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T
#    EnsTime = (Ntimesteps) numpy vector.
def SIRens(ParamArray, Tstart, Tstop, Ntimestep):
    # ParamArray = (4)x(EnSize) Numpy array of ensemble member parameters
    #              <S0>\t<I0>\t<beta>\t<mu>
    # Tstart = start time
    # Tstop = stop time
    # Ntimestep = number of time steps
    EnSize = ParamArray.shape[1]

    # Define time 
    EnsTime = np.linspace(Tstart,Tstop,Ntimestep)

    # Define empty array to append ensemble members to
    EnsArray = np.zeros((2*Ntimestep,EnSize))

    # Generate each of the ensemble members
    for i in range(EnSize):
        # Read file ICs to variable names
        S0 = ParamArray[0,i]
        I0 = ParamArray[1,i]
        beta = ParamArray[2,i]
        mu = ParamArray[3,i]

        # Define inputs to SDEsim() from ICs
        y0 = np.array([[S0],[I0]])
 
        # Simulate SIR
        Xsim = epiODElib.SIRode(y0, EnsTime, beta, mu)
        Xsim = Xsim.transpose()

        # Reshape and write to EnsArray.
        EnsArray[:,i] = Xsim.reshape(2*Ntimestep)

    return [EnsArray, EnsTime]

##################################################################
def SIRens_sde(ParamArray, Tstart, Tstop, Ntimestep):
    # SIR ensemble except with stochastic differential equation
    # ParamArray = (4)x(EnSize) Numpy array of ensemble member parameters
    #              <S0>\t<I0>\t<beta>\t<mu>
    # Tstart = start time
    # Tstop = stop time
    # Ntimestep = number of time steps
    EnSize = ParamArray.shape[1]

    # Define time 
    EnsTime = np.linspace(Tstart,Tstop,Ntimestep)

    # Define empty array to append ensemble members to
    EnsArray = np.zeros((2*Ntimestep,EnSize))

    # Generate each of the ensemble members
    for i in range(EnSize):
        # Read file ICs to variable names
        S0 = ParamArray[0,i]
        I0 = ParamArray[1,i]
        beta = ParamArray[2,i]
        mu = ParamArray[3,i]

        # Define inputs to SDEsim() from ICs
        y0 = np.array([[S0],[I0]])
 
        # Simulate SDE-SIR
        Nstep = 10 # Number stochastic steps per timestep
        sigma = 0. # No model error term here
        Xsim = epiODElib.SIRsde_noise(y0,EnsTime,Nstep,beta,mu,sigma)
        Xsim = Xsim.transpose()

        # Reshape and write to EnsArray.
        EnsArray[:,i] = Xsim.reshape(2*Ntimestep)

    return [EnsArray, EnsTime]

##################################################################
# Function to map ensemble to measurements. Measurements 
# returned is in form to be passed to an analysis.
#
# RETURNS:
#      EnsObs = (1)x(EnSize) numpy vector of infected counts at 
#               measurement time.
def SIRmeasure(EnsArray):
    # EnsArray = (2*Ntimestep)x(EnSize) numpy array. Column is 
    #            (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T

    EnsObs = EnsArray[-1]
    return EnsObs

# For ILI data must multiply by 100%
def ILImeasure(EnsArray):
    # EnsArray = (2*Ntimestep)x(EnSize) numpy array. Column is 
    #            (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T

    EnsObs = 100.0*EnsArray[-1]
    return EnsObs
##################################################################
# RETURNS:
#      EnsData = (Ntimestep)x(2*EnSize+1) numpy array of SIR 
#                ensemble simulations. Has format,
#      t0 S1(t0) I1(t0) . . . Sm(t0) Im(t0)
#      t1 S1(t1) I1(t1) . . . Sm(t1) Im(t1)
#       .   .      .            .      .
#       .   .      .            .      .
#       .   .      .            .      .
#      tN S1(tN) I1(tN) . . . Sm(tN) Im(tN)
def SIRshape(EnsArray,EnsTime):
    #    EnsArray = (2*Ntimestep)x(EnSize) numpy array. Column is 
    #               (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T
    #    EnsTime = (Ntimesteps) numpy vector.
    EnSize = EnsArray.shape[1]
    Ntimestep = EnsTime.shape[0]

    EnsData = np.zeros((Ntimestep, 2*EnSize+1))
                       
    EnsData[:,0] = EnsTime

    for i in range(EnSize):
        EnsData[:,(2*i+1):(2*i+3)] = EnsArray[:,i].reshape(Ntimestep,2) 

    return EnsData
##################################################################

