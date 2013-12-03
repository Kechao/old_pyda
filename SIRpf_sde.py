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

# This script performs ensemble generation and analysis generation for SIR data using the 
# most naive version of particle filtering (PF). Weighting is done using the 'forward likelihood'
# as an importance sampler and resampling is done just using the weights. The ensemble is 
# generated stochastically so the assimilation is less degenerate. Parameter exploration is 
# still highly degenerate. Propagation is only from current data point. However, due to the nature
# of the filter-resample scheme balances are maintained.

# Use:
#    SIRpf_sde.py <DataFileName> <DataNoise> <Tspan>

# To use this the ensemble parameter draws from the prior
# distributions must be generated already:
#     ./Data/params.0.dat

import sys
import numpy as np 
import numpy.random as rn
import math 

# User defined package
import pflib
import SIR_ensemble_lib

###################################################################
#                           MAIN CODE                             #
###################################################################
# Load data and pickout number of measurements to assimilate
DataArray = np.loadtxt(sys.argv[1],delimiter='\t',skiprows=1)
Ndata_pts = DataArray.shape[0]

# Estimated noise in data
DataNoise = float(sys.argv[2])

# Get end simulation time
Tspan = float(sys.argv[3])

# Set start time for ensemble to 0
Tstart = 0.0

print 'Data Time: ', Tstart

# Define resolution of SIR simulation:
# Per ensemble interval.
Ntimestep = 50.
Ncaststep = 1500.

# Create ensemble and perform analysis iteratively 
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

    # Form data covariance matrix
    DatCov = np.zeros((1,1))
    DatCov[0] = math.pow(DataNoise,2)

    # Pull out data array and make it size of ensemble.
    Data = DataArray[i,2]*np.ones((1,EnSize))

    # Generate the ensemble
    [EnsArray,EnsTime] = SIR_ensemble_lib.SIRens_sde(ParamArray,Tstart,Tstop,Ntimestep)

    # Generate ensemble forecast
    [EnsCast,EnCastTime] = SIR_ensemble_lib.SIRens_sde(ParamArray,Tstart,Tspan, 
                                                       Ncaststep-i*Ntimestep)

    # Generate observations from the ensemble.
    EnsObs = SIR_ensemble_lib.SIRmeasure(EnsArray)    

    # Calculate particle weights
    Weights = pflib.naivePF(Data,DatCov,EnsArray,ParamArray,EnsObs)

    # Generate analysis by resampling
    [AnalysisEnsemble,AnalysisParams] = pflib.naiveResamp(EnsArray,ParamArray,Weights)

    # Update the parameter array to represent (S(t), I(t)) at the 
    # measurement time to use in the next ensemble generation.
    AnalysisParams[:2,:] = AnalysisEnsemble[-2:,:]

    # Generate analyzed ensemble forecast
    [AnalysisCast,AnCastTime] = SIR_ensemble_lib.SIRens_sde(AnalysisParams,Tstop,Tspan,
                                                            Ncaststep-i*Ntimestep)

    # Put parameter array in proper form for writing to file.
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

    #################
    #     UPDATE    #
    #################
    Tstart = Tstop
    print 'Data Time: ', Tstart

