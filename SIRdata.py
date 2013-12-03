#!/usr/bin/python

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

# Use:
# SIRdata.py <dataICname> <SimResultname> <DataResultname>

# Generate data with noise from an Epidemic simulation. 
#
# In the input file (dataICname) are the data generation specifics:
#       - parameter set ([beta,mu,I0] for SIR)
#       - time interval 
#       - number of data points
#       - data noise level

# Possible Epidemic simulations are in epiODElib.py.
#######################################################################
#######################################################################
# Import necessary modules.
import sys
import numpy as np 
import numpy.random as rn

# User defined package
import epiODElib
#######################################################################
#######################################################################
# Open Data ICs file for reading
dataICfile = open(sys.argv[1], 'r')
dataIC = dataICfile.readlines()
dataIClist = dataIC[1].split()
dataICfile.close()

# Read file ICs to variable names
I0 = float(dataIClist[0])
Tspan = float(dataIClist[1])
Ndata_pts = float(dataIClist[2])
beta = float(dataIClist[3])
mu = float(dataIClist[4])
noise = float(dataIClist[5])

#####################################################
# Define inputs to epiODE simulation from ICs
S0 = 1.0 - I0
y0 = np.array([[S0],[I0]])

# Define resolution of simulation.
# Number of deterministic steps.
# NOTE: THIS SHOULD BE SPECIFIED IN dataIC.dat
Ntimesteps = 1500.

# Use simulation resolution to define number of steps 
# between measurements.
Data_Increment = int(Ntimesteps/Ndata_pts)

# Define time vector
time = np.linspace(0.,Tspan,Ntimesteps)

# Simulate SIR
Xsim = epiODElib.SIRode(y0, time, beta, mu)

# Subsample and add noise to generate data
Data = Xsim[:,Data_Increment::Data_Increment] 
DataLength = Data.shape[1]

# Define measurement time vector
measured_time = time[Data_Increment::Data_Increment]

# Make sure number of data points and time instances match
measured_time = measured_time[range(DataLength)]

Data = Data + noise*rn.randn(2,DataLength)

###########################################################
# Write simulation and data to files with headers
# First write simulation result to file
# Simulation Header String
SimHeader = '<time>\t<Susceptible>\t<Infected>'

# Write simulation to array
SimArray = np.vstack([time, Xsim])
SimArray = SimArray.T

# Write simulation results
np.savetxt(sys.argv[2],SimArray,fmt='%5.5f',delimiter='\t',
           header=SimHeader)

# Second, write the data to file
# Data Header String
DataHeader = '<time>\t<Susceptible>\t<Infected>'

# Write data to array
DataArray = np.vstack([measured_time, Data])
DataArray = DataArray.T

# Write data results
np.savetxt(sys.argv[3],DataArray,fmt='%5.5f',delimiter='\t',
           header=DataHeader)



