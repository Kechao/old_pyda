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
# AssimCone.py <SimRunName> <DataName> <EnsNumber>

# Uses filled confidence curves to plot results of simulation, data
# collection, ensemble, and analysis in two graphs. One for the
# ensemble array and one for the analysis array.

import sys
import numpy as np 
import scipy.stats.mstats as mst
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

# Define ensemble number to plot.
EnsNumber = int(sys.argv[3])

# Define quantiles
QntLev = [0.10, 0.5, 0.9]

##########################
CurFig = plt.figure()
# Ensemble setup
EnsembleFileName = "".join(('./data/ensemble.',str(EnsNumber),'.dat'))
EnsembleArray = np.loadtxt(EnsembleFileName)
    
# Grab ensemble time
ensembletime = EnsembleArray[:,0]

# Pull array of susceptible ensemble runs and infected ensemble runs
EnSus = EnsembleArray[:,1::2]
EnInf = EnsembleArray[:,2::2]
EnSize = EnSus.shape[1]

# Calculate quantile curves
EnSusQnt = mst.mquantiles(EnSus,prob=QntLev,axis=1)
EnInfQnt = mst.mquantiles(EnInf,prob=QntLev,axis=1)

# Analysis setup
AnalysisFileName = "".join(('./data/analysis.',str(EnsNumber),'.dat'))
AnalysisArray = np.loadtxt(AnalysisFileName)
    
# Grab analysis time
analysistime = AnalysisArray[:,0]

# Pull array of susceptible analysis runs and infected analysis runs
AnSus = AnalysisArray[:,1::2]
AnInf = AnalysisArray[:,2::2]
AnSize = AnSus.shape[1]

# Calculate quantile curves
AnSusQnt = mst.mquantiles(AnSus,prob=QntLev,axis=1)
AnInfQnt = mst.mquantiles(AnInf,prob=QntLev,axis=1)

# Plot confidence cone
plt.subplot(211)
EnShd = plt.fill_between(ensembletime,EnSusQnt[:,0],EnSusQnt[:,2],
                         color=(66./255.,253./255.,234./255.),alpha=0.3)
AnShd = plt.fill_between(analysistime,AnSusQnt[:,0],AnSusQnt[:,2],
                         color=(249./255.,138./255.,168./255.),alpha=0.1)
plt.subplot(212)
plt.fill_between(ensembletime,EnInfQnt[:,0],EnInfQnt[:,2],
                 color=(66./255.,253./255.,234./255.),alpha=0.3)
plt.fill_between(analysistime,AnInfQnt[:,0],AnInfQnt[:,2],
                 color=(249./255.,138./255.,168./255.),alpha=0.1)

# Plot median
plt.subplot(211)
EnMed = plt.plot(ensembletime,EnSusQnt[:,1],'b-',linewidth=2)
AnMed = plt.plot(analysistime,AnSusQnt[:,1],'g-',linewidth=2)
plt.subplot(212)
plt.plot(ensembletime,EnInfQnt[:,1],'b-',linewidth=2)
plt.plot(analysistime,AnInfQnt[:,1],'g-',linewidth=2)

# Plot simulation and data over ensemble plot
plt.subplot(211)
line1 = plt.plot(simtime, SimSus)
plt.setp(line1,linewidth=2,color='r')
Mk1 = plt.plot(datatime[0:(EnsNumber+1)], DataSus[0:(EnsNumber+1)], 'bo')
plt.ylabel('Susceptible')
plt.title('SIR Data')

plt.subplot(212)
line2 = plt.plot(simtime, SimInf)
plt.setp(line2,linewidth=2,color='r')
plt.plot(datatime[0:(EnsNumber+1)], DataInf[0:(EnsNumber+1)], 'yD')
plt.xlabel('Time')
plt.ylabel('Infected')
  
# Save figure  
FigName = "".join(('./figures/SIR_AssimilationCone.',str(EnsNumber),'.eps'))
plt.savefig(FigName)



