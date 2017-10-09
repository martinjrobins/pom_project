#!/bin/bash
module load cmake/3.8.0
module load gcc/4.8.2
#module load gcc/5.4.0
#module load vtk
#module load openmpi/1.8.4__gcc-4.9.2
#module load gpu/cuda/8.0.44
module load python
#module load python/2.7__gcc-5.3__graph_tools
#module load gpu/cuda/7.5.18
export BOOST_ROOT=/system/software/linux-x86_64/lib/boost/1_60_0
export EIGEN_ROOT=/system/software/linux-x86_64/lib/eigen/3.2.8/
virtualenv env
source env/bin/activate
#export set PYTHONPATH=../pints:../pints/problems/electrochemistry
pip install cma numpy matplotlib scipy

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBoost_NO_SYSTEM_PATHS=BOOL:ON \
    -DBoost_NO_BOOST_CMAKE=BOOL:ON \
    -DBOOST_LIBRARYDIR=$BOOST_ROOT/lib \
    -DBOOST_INCLUDEDIR=$BOOST_ROOT/include \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_ROOT/include/eigen3 \
    ../pints/problems/electrochemistry


