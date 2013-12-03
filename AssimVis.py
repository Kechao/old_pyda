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
# AssimVis.py <SimRunName> <DataName>

# Plot results of simulation, data collection, ensemble, and analysis 
# on one graph.

import sys
import numpy as np 
import scipy.linalg as linalg
import numpy.random as rn
import math 
import matplotlib.pyplot as plt 

# First load respective files into NumPy arrays and set ensemble size
SimArray = np.loadtxt(sys.argv[1],delimiter='\t',skiprows=1)
DataArray = np.loadtxt(sys.argv[2],delimiter='\t',skiprows=1)
Ndata_pts = DataArray.shape[0]

# Pull out respective simulation and data time arrays
simtime = SimArray[:,0]
datatime = DataArray[:,0]

# Pull out simulation and data susceptible and infected timeseries
SimSus = SimArray[:,1]
DataSus = DataArray[:,1]

SimInf = SimArray[:,2]
DataInf = DataArray[:,2]

# Plot ensemble runs first
for k in range(Ndata_pts):
    plt.figure(k+1)
    EnsembleFileName = "".join(('./data/ensemble.',str(k),'.dat'))
    EnsembleArray = np.loadtxt(EnsembleFileName)

    # Grab ensemble time
    ensembletime = EnsembleArray[:,0]

    # Pull array of susceptible ensemble runs and infected ensemble runs
    EnSus = EnsembleArray[:,1::2]
    EnInf = EnsembleArray[:,2::2]
    EnSize = EnSus.shape[1]

    for i in range(EnSize):
        # Light Blue: color=(36./255.,164./255.,239./255.)
        plt.subplot(211)
        plt.plot(ensembletime,EnSus[:,i],color=(36./255.,164./255.,239./255.),linewidth=.15)
        plt.subplot(212)
        plt.plot(ensembletime,EnInf[:,i],color=(36./255.,164./255.,239./255.),linewidth=.15)

    # Plot analysis ensemble second
    AnalysisFileName = "".join(('./data/analysis.',str(k),'.dat'))
    AnalysisArray = np.loadtxt(AnalysisFileName)

    # Grab analysis time
    analysistime = AnalysisArray[:,0]

    # Pull array of susceptible ensemble runs and infected ensemble runs
    AnSus = AnalysisArray[:,1::2]
    AnInf = AnalysisArray[:,2::2]
    AnSize = AnSus.shape[1]

    for i in range(AnSize):
        plt.subplot(211)
        plt.plot(analysistime,AnSus[:,i],'g-',linewidth=.15)
        plt.subplot(212)
        plt.plot(analysistime,AnInf[:,i],'g-',linewidth=.15)

    # Plot simulation and data over ensemble plot
    plt.subplot(211)
    line1 = plt.plot(simtime, SimSus)
    plt.setp(line1,linewidth=2,color='r')
    plt.ylabel('Susceptible')
    plt.title('SIR Data Assimilation, SIRkf')

    plt.subplot(212)
    line2 = plt.plot(simtime, SimInf)
    plt.setp(line2,linewidth=2,color='r')
    # All Data
    plt.plot(datatime[0:(k+1)], DataInf[0:(k+1)], 'yD')

    # # Last data point separate
    # plt.plot(datatime[0:(k-1)], DataInf[0:(k-1)], 'yD')
    # plt.plot(datatime[(k-1):(k+1)], DataInf[(k-1):(k+1)], 'ro')

    # # Last two data points separate
    # plt.plot(datatime[0:(k-2)], DataInf[0:(k-2)], 'yD')
    # plt.plot(datatime[(k-2):(k)], DataInf[(k-2):(k)], 'ro')

    plt.xlabel('Time')
    plt.ylabel('Infected')
    # Turn plot ticks/numbering off
    # plt.tick_params(axis='x', which='both',bottom='off',top='off',labelbottom='off')
    # plt.tick_params(axis='y', which='both',left='off',right='off',labelleft='off')
    # plt.ylim([0,1]) # If you want to fix y-axis
    # plt.show()

    FigName = "".join(('./figures/SIR_assimilation.',str(k),'.eps'))
    plt.savefig(FigName)

