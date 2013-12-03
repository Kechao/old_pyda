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

######################################################################
# Library of functions for PF based ensemble analysis. For the Particle 
# filtering methods the goal is to return a weight associatted with each 
# particle in the ensemble. Then one must resample using one of the 
# resampling functions provided. 
######################################################################
import numpy as np 
import scipy.linalg as linalg
import numpy.random as rn
import math 

######################################################################
# All particle filter functions:
# INPUT: {numpy arrays}
#      <Data Array> = (measurement size)x(ensemble size)
#      <Data Covariance> = (measurement size)x(measurement size)
#      <Ensemble Array> = (simulation size)x(ensemble size)
#      <Parameter Array> = (parameter size)x(ensemble size)
#      <Ensemble Observation Array> = (measurement size)x(ensemble size)
#
# RETURN: {numpy arrays}
#      <Particle Weight Array> = (1)x(ensemble size)
######################################################################

# Function to generate PF analysis. 
# In this, most basic, form of the PF the Param array, and Ensemble 
# array are not used. Particle weights are only determined by the Data, 
# DatCov, and the EnsObs associatted with each particle. The data likelihood 
# is assumed to be Gaussian. 
def naivePF(Data,DatCov,Ensemble,Param,EnsObs):
    # Collect data sizes.
    EnSize = Ensemble.shape[1]
    MeaSize = Data.shape[0]

    # First create weight array.
    # W is (1)x(EnSize)
    W = np.zeros(EnSize)

    # Calculate data perturbations from ensemble measurements
    # Dpert = (MeasSize)x(EnSize)
    Dpert = Data - EnsObs

    # Compute inv(DataCov)*Dpert
    # Should be (MeasSize)x(EnSize)
    B = linalg.solve(DatCov,Dpert)

    # Calculate un-normalized weight for each particle using observations
    NormArg = np.diag(np.dot(Dpert.transpose(),B))
    W = np.exp(-(0.5)*(NormArg))

    # Now normalize weights
    W = W/np.sum(W)

    return W

######################################################################
# Resampling functions use the Ensemble and Parameter arrays, along with 
# weights calulated with one of the Particle filters, to generate an 
# analysis Ensemble with equal weights.
######################################################################

######################################################################
# All resampling functions:
# INPUT: {numpy arrays}
#      <Ensemble Array> = (simulation size)x(ensemble size)
#      <Parameter Array> = (parameter size)x(ensemble size)
#      <Particle Weight Array> = (1)x(ensemble size)
#
# RETURN: {numpy arrays}
#      <Analysis Array> = (simulation size)x(ensemble size)
#      <Analysis Parameter Array> = (parameter size)x(ensemble size)
######################################################################

######################################################################
# Function to generate PF resampled analysis ensemble. 
# In this, most basic, form of resampling ensemble members are resampled 
# according to their weights directly. This will cause problematic duplications
# in parameter samples.
def naiveResamp(Ensemble,Param,Weight):
    # Get ensemble size
    EnSize = Ensemble.shape[1]

    # Generate resampled indices
    index = range(EnSize)
    resamp_index = rn.choice(index,size=EnSize,replace=True,p=Weight)

    # Create analysis ensembles
    AnalysisEnsemble = Ensemble[:][:,resamp_index] 
    AnalysisParams = Param[:][:,resamp_index]

    return [AnalysisEnsemble,AnalysisParams]

