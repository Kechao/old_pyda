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

# Write prior ensemble parameters to file
# <S0>\t<I0>\t<beta>\t<mu>
# Use:
#  EnParamGen.py <ParamsOutputName> <EnsembleSize>

# Write ensemble parameters to use to a file
import sys
import numpy as np 
import random as rn
import math 

##############################################
# Distributed, imprecisely known, parameters #
##############################################
# # Prior distributions are Log-Normal specified by mean and variance
# I0_mean_sqr = math.pow(10.0,2) 
# I0_var = 3.0
# # Shape is mean of associated normal distribution
# I0_shp = math.log(I0_mean_sqr/math.sqrt(I0_var+I0_mean_sqr))
# # Scale is standard deviation of associated normal distribution
# I0_scl = math.sqrt(math.log(1+I0_var/I0_mean_sqr))

# beta_mean_sqr = math.pow(0.1,2) 
# beta_var = 0.02
# # Shape is mean of associated normal distribution
# beta_shp = math.log(beta_mean_sqr/math.sqrt(beta_var+beta_mean_sqr))
# # Scale is standard deviation of associated normal distribution
# beta_scl = math.sqrt(math.log(1+beta_var/beta_mean_sqr))

# mu_mean_sqr = math.pow(0.05,2) 
# mu_var = 0.0125
# # Shape is mean of associated normal distribution
# mu_shp = math.log(mu_mean_sqr/math.sqrt(mu_var+mu_mean_sqr))
# # Scale is standard deviation of associated normal distribution
# mu_scl = math.sqrt(math.log(1+mu_var/mu_mean_sqr))

# #########################
# # Prior distributions are Log-Normal specified by median and mode. This
# # makes it easier to control/ensure centering.
# I0_mode = 10.0
# I0_med = 12.0
# # Shape is mean of associated normal distribution
# I0_shp = math.log(I0_med)
# # Scale is standard deviation of associated normal distribution
# I0_scl = math.sqrt(math.log(I0_med/I0_mode))

# beta_mode = 0.1
# beta_med = 0.15
# # Shape is mean of associated normal distribution
# beta_shp = math.log(beta_med)
# # Scale is standard deviation of associated normal distribution
# beta_scl = math.sqrt(math.log(beta_med/beta_mode))

# mu_mode = 0.05
# mu_med = 0.055
# # Shape is mean of associated normal distribution
# mu_shp = math.log(mu_med)
# # Scale is standard deviation of associated normal distribution
# mu_scl = math.sqrt(math.log(mu_med/mu_mode))

# Prior distributions are centered uniform distribution of specified width.
I0center = 0.01
I0width = 0.005

betacenter = 0.1
betawidth = 0.05

mucenter = 0.05
muwidth = 0.025
##############################################

EnSize = float(sys.argv[2])
I0 = np.zeros((EnSize))
beta = np.zeros((EnSize))
mu = np.zeros((EnSize))
for i in range(int(EnSize)):
    # # When using Log-Normal priors
    # I0[i] = rn.lognormvariate(I0_shp,I0_scl)
    # beta[i] = rn.lognormvariate(beta_shp,beta_scl)
    # mu[i] = rn.lognormvariate(mu_shp,mu_scl)
    
    # # When using uniform priors
    I0[i] = rn.uniform(I0center-(I0width/2.0),I0center+(I0width/2.0))
    beta[i] = rn.uniform(betacenter-(betawidth/2.0),betacenter+(betawidth/2.0))
    mu[i] = rn.uniform(mucenter-(muwidth/2.0),mucenter+(muwidth/2.0))

# Technically I0 could be larger than 1.0 (more than 100% of population infected) 
# if using Log-Normal priors.
I0 = np.minimum(I0,1.0)

# Initial Susceptible 
S0 = np.ones((EnSize)) - I0

# Write prior parameter samples.
# Combine parameters into array.
ParamArray = np.vstack([S0,I0,beta,mu])
ParamArray = ParamArray.T
    
# Header string
ParHeader = '<S0>\t<I0>\t<beta>\t<mu>'

# Write parameter array
np.savetxt(sys.argv[1],ParamArray,fmt='%5.5f',delimiter='\t',
           header=ParHeader)

