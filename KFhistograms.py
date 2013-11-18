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

# Script to plot distributions of SIR parameters as new data is
# assimilated.

# USE:
#     KFhistograms.py 

import numpy as np
import matplotlib.pyplot as plt
################################################################
# Indices of parameter sets to examine
ParIndex = [0,3,6,9]

# True parameter values
I0 = 10.
beta = 0.1
mu = 0.05

# Number of bins
num_bins = 10

cnt = 0

plt.figure(1)
for i in ParIndex:    
    # Open parameter set file and get array
    # <S0>\t<I0>\t<beta>\t<mu>
    paramfile = "".join(('./data/SIRkf4/params.',str(i),'.dat'))
    ParamArray = np.loadtxt(paramfile,delimiter='\t',skiprows=1)
    ParamArray = ParamArray.transpose()

    # Histogram of I0
    plt.subplot(4,3,3*cnt+1)
    plt.hist(ParamArray[1,:],num_bins,normed=True,facecolor='blue')
    plt.vlines(I0,0.,1.,'r')
    plt.xlabel('I0')
    plt.ylabel('Prob.')

    # Histogram of Beta
    plt.subplot(4,3,3*cnt+2)
    plt.hist(ParamArray[2,:],num_bins,normed=True,facecolor='blue')
    plt.vlines(beta,0.,10.,'r')
    plt.xlabel('Trans. Rt.')
    plt.ylabel('Prob.')

    # Histogram of mu
    plt.subplot(4,3,3*cnt+3)
    plt.hist(ParamArray[3,:],num_bins,normed=True,facecolor='blue')
    plt.vlines(mu,0.,10.,'r')
    plt.xlabel('Rec. Rt.')
    plt.ylabel('Prob.')
    
    cnt += 1

FigName = './figures/histSIRkf4.eps'
plt.savefig(FigName)
