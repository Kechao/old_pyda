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

import numpy as np 
import scipy.linalg as linalg
import numpy.random as rn
import math 

######################################################################
# Library of functions for KF based ensemble analysis.

# All functions:
# INPUT: {numpy arrays}
#      <Data Array> = (measurement size)x(ensemble size)
#      <Data Covariance> = (measurement size)x(measurement size)
#      <Ensemble Array> = (simulation size)x(ensemble size)
#      <Parameter Array> = (parameter size)x(ensemble size)
#      <Ensemble Observation Array> = (measurement size)x(ensemble size)
#
# RETURN: {numpy arrays}
#      <Analysis Array> = (simulation size)x(ensemble size)
#      <Analysis Parameter Array> = (parameter size)x(ensemble size)
######################################################################

# Function to generate enKF analysis. 
# In this, most basic, form of the enKF the analysis covariance is not
# computed explicitly. Instead, each ensemble member is updated individually. 
def enKF1(Data,DatCov,Ensemble,Param,EnsObs):
    # Collect data sizes.
    EnSize = Ensemble.shape[1]
    SimSize = Ensemble.shape[0] 
    MeaSize = Data.shape[0]

    # First combine the Ensemble and Param arrays.
    # A is (SimSize+ParSize)x(EnSize)
    A = np.vstack([Ensemble, Param])

    # Calculate mean
    Amean = (1./float(EnSize))*np.tile(A.sum(1), (EnSize,1)).transpose()

    # Calculate ensemble perturbation from mean
    # Apert should be (SimSize+ParSize)x(EnSize)
    Apert = A - Amean

    # Data perturbation from ensemble measurements
    # Dpert should be (MeasSize)x(EnSize)
    Dpert = Data - EnsObs

    # Ensemble measurement perturbation from ensemble measurement mean.
    # This is a poor approximation of required S if measurement operator is 
    # non-linear.
    # S is (MeasSize)x(EnSize)
    MeasAvg = (1./float(EnSize))*np.tile(EnsObs.reshape(MeaSize,EnSize).sum(1), 
                                         (EnSize,1)).transpose()
    S = EnsObs - MeasAvg

    # Set up covariance scalar
    # COV is (MeasSize)x(MeasSize)
    COV = (1./float(EnSize-1))*np.dot(S,S.transpose()) + DatCov
    
    # Compute inv(COV)*Dpert
    # Should be (MeasSize)x(EnSize)
    B = linalg.solve(COV,Dpert)

    # Adjust ensemble perturbations
    # Should be (SimSize+ParSize)x(MeasSize)
    ApertS = (1./float(EnSize-1))*np.dot(Apert,S.transpose())

    # Compute analysis
    # Analysis is (SimSize+ParSize)x(EnSize)
    Analysis = A + np.dot(ApertS,B)

    # Pull off Analyzed Ensemble and Analyzed Parameters.
    AnalysisEnsemble = Analysis[0:SimSize,:]
    AnalysisParams = Analysis[SimSize:,:]

    return [AnalysisEnsemble,AnalysisParams]

