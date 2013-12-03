#!/bin/bash

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

LOC_DIR='./pyDA'

echo "Creating synthetic data."
./SIRdata.py './data/dataIC.dat' './data/simSIR.dat' './data/simSIRdata.dat'

echo "Creating initial parameter file."
./EnParamGen.py './data/params.0.dat' 100.0

echo "Generating forecasts and analysis."
./SIRkf1.py './data/simSIRdata.dat' 0.0025 200.0 

echo "Plotting the assimilation process."
./AssimVis.py './data/simSIR.dat' './data/simSIRdata.dat'

./AssimCone.py './data/simSIR.dat' './data/simSIRdata.dat' 5