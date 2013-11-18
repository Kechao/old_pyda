#!/usr/bin/python

###############################################################################
###############################################################################
#   Copyright 2013 Kyle S. Hickmann                                           
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

# This script performs ensemble generation and analysis generation for
# SIR data. It uses re-propagation through the SIR model starting at
# t0 to maintain SIR balances during assimilation. Analysis forecasts
# are done starting at t0. Unlike the assimilation performed in
# SIRkf4.py, here the assimilation step uses only the most recent 3
# measured data points, including the current data point. Ensemble
# measurements are adjusted accordingly. This most closely resembles
# Evensen's Ensemble Kalman Smoother.

# Use:
#    SIRkf5.py <DataFileName> <DataNoise> <Tspan>

# To use this the ensemble parameter draws from the prior
# distributions must be generated already:
#     ./Data/params.0.dat

import sys
import numpy as np 
import numpy.random as rn
import math 

# User defined package
import kflib
import SIR_ensemble_lib

###################################################################
###################################################################
#                           MAIN CODE                             #
###################################################################
###################################################################
#                           MAIN CODE                             #
###################################################################
###################################################################
# Load data and pickout number of measurements to assimilate
DataArray = np.loadtxt(sys.argv[1],delimiter='\t',skiprows=1)
Ndata_pts = DataArray.shape[0]

# Estimated noise in data
DataNoise = float(sys.argv[2])

# Get end simulation time
Tspan = float(sys.argv[3])

# Set start time for ensemble to 0
Tinit = 0.0

print 'Data Time: ', Tinit

# Define resolution of SIR simulation:
# Per ensemble interval.
Ntimestep = 50.
Ncaststep = 1500.

# Max number of data points to assimilate at one time.
AssimSize = 3

# Create ensemble and perform analysis iteratively 
# Only using 9 data points.
for i in range(Ndata_pts):
    #####################
    # GENERATE ENSEMBLE # 
    #####################
    # Open ensemble parameters file for reading
    # <S0>\t<I0>\t<beta>\t<mu>
    paramfile = "".join(('./data/params.',str(i),'.dat'))
    ParamArray = np.loadtxt(paramfile,delimiter='\t',skiprows=1)
    ParamArray = ParamArray.transpose()
    EnSize = ParamArray.shape[1]

    # Define stop time for ensemble run
    Tstop = DataArray[i,0]

    DataIndex = min(i+1,AssimSize)

    # Form data covariance matrix
    DatCov = np.diag(math.pow(DataNoise,2)*np.ones((DataIndex)),0)

    # Pull out data array and add random perturbations.
    Data = np.tile(DataArray[(i-(DataIndex-1)):(i+1),2],(EnSize,1)).transpose()\
           + DataNoise*rn.randn(DataIndex,EnSize)

    # Generate ensemble forecast
    [EnsCast, EnCastTime] = SIR_ensemble_lib.SIRens(ParamArray,Tinit,Tspan, 
                                                    Ncaststep)

    # Generate the ensemble and measurements up until current measurement
    for j in range(DataIndex):
        [EnsArray, EnsTime] = SIR_ensemble_lib.SIRens(ParamArray,Tinit,DataArray[i-j,0],
                                                      (i-j+1)*Ntimestep)

        # Generate observations from the ensemble.
        # If first observation then create observation array.
        # Otherwise append new observation to previous.
        if (j == 0):
            EnsObs = SIR_ensemble_lib.SIRmeasure(EnsArray)    
        else:
            EnsObs = np.vstack([SIR_ensemble_lib.SIRmeasure(EnsArray),EnsObs])

    # Perform analysis step
    [AnalysisEnsemble,AnalysisParams] = kflib.enKF1(Data,DatCov,EnsArray,ParamArray,
                                                    EnsObs)

    # Since enKF assumes Gaussianity we must adjust the parameters first to 
    # maintain positivity. This still could allow starting S0+I0 > 1.0.
    AnalysisParams = np.maximum(AnalysisParams,0.0)
    
    # At this point we could have S0+I0 != 1.0 in AnalysisParams.
    # We fix this here by setting S0 = 1.0-I0.
    AnalysisParams[0,:] = 1.0 - AnalysisParams[1,:]

    # Generate analyzed ensemble forecast
    [AnalysisCast,AnCastTime] = SIR_ensemble_lib.SIRens(AnalysisParams,Tinit,Tspan,
                                                        Ncaststep)

    # Put Analyzed Parameters in correct form for writing to file.
    AnalysisParams = AnalysisParams.transpose()

    #########
    # Write analyzed parameters to be used in next ensemble generation.
    AnalysisParName = "".join(('./data/params.',str(i+1),'.dat'))
    
    # Header string
    ParHeader = '<S0>\t<I0>\t<beta>\t<mu>'

    # Write parameter array
    np.savetxt(AnalysisParName,AnalysisParams,fmt='%5.5f',delimiter='\t',
               header=ParHeader)

    #########
    # Write ensemble forecast to observe prior exploration of prediction
    # space.
    EnsCastFileName = "".join(('./data/ensemble.',str(i),'.dat'))
    EnsembleCastWrite = SIR_ensemble_lib.SIRshape(EnsCast,EnCastTime)
    np.savetxt(EnsCastFileName,EnsembleCastWrite,fmt='%5.5f',delimiter=' ')

    #########
    # Write analysis forecast to observe posterior exploration of 
    # prediction space.
    AnalysisCastFileName = "".join(('./data/analysis.',str(i),'.dat'))
    AnalysisCastWrite = SIR_ensemble_lib.SIRshape(AnalysisCast,AnCastTime)
    np.savetxt(AnalysisCastFileName,AnalysisCastWrite,fmt='%5.5f',delimiter=' ')

    # Print timestep just assimilated
    print 'Data Time: ', Tstop

