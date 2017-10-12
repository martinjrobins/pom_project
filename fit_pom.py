import sys
sys.path.insert(0, 'pints')
sys.path.insert(0, 'pints/problems/electrochemistry')
sys.path.insert(0, 'build')
import pints
import electrochemistry
import pickle
import numpy as np
import os
import sys
import math

import matplotlib as mpl
#mpl.use('Agg')
import pylab as plt
import numpy.fft as fft
from math import pi

filename = 'POMGCL_6020104_1.0M_SFR_d16.txt'

diff_i = int(sys.argv[1])
sample_i = int(sys.argv[2])
reversible = sys.argv[3].lower() == 'true'
if reversible:
   print 'REVERSIBLE'
else:
   print 'QUASI-REVERSIBLE'


dim_params = {
    'reversed': False,
    'Estart': 0.6,
    'Ereverse': -0.1,
    'omega': 60.05168,
    'phase': 0,
    'dE': 20e-3,
    'v': -0.1043081,
    't_0': 0.00,
    'T': 298.2,
    'a': 0.0707,
    'c_inf': 0.1*1e-3*1e-3,
    'Ru': 50.0,
    'Cdl': 0.000008,
    'Gamma' : 0.7*53.0e-12,
    'alpha1': 0.5,
    'alpha2': 0.5,
    'E01': 0.368,
    'E02': 0.338,
    'E11': 0.227,
    'E12': 0.227,
    'E21': 0.011,
    'E22': -0.016,
    'k01': 1e4,
    'k02': 1e4,
    'k11': 1e4,
    'k12': 1e4,
    'k21': 1e4,
    'k22': 1e4
    }


poms_model = electrochemistry.POMModel(dim_params)
data = electrochemistry.ECTimeData(filename,poms_model,ignore_begin_samples=5,ignore_end_samples=0)

if reversible:
   names = ['E01',
         'E02',
         'E11',
         'E12',
         'E21',
         'E22',
         'gamma']
else:
   names = ['E01',
         'E02',
         'E11',
         'E12',
         'E21',
         'E22',
         'k01',
         'k02',
         'k11',
         'k12',
         'k21',
         'k22',
         'gamma']

e0_buffer = 0.1*(poms_model.params['Estart'] - poms_model.params['Ereverse'])
max_current = np.max(data.current)
max_k0 = poms_model.non_dimensionalise(10000,'k01')

if reversible:
   lower_bounds = [poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                0.1,
                0.005*max_current]

   upper_bounds = [poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                5,
                0.03*max_current]
else:

   lower_bounds = [poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0.005*max_current]

   upper_bounds = [poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                5,
                0.03*max_current]

priors = []
E0 = 0.5*(poms_model.params['E01'] + poms_model.params['E02'])
E1 = 0.5*(poms_model.params['E11'] + poms_model.params['E12'])
E2 = 0.5*(poms_model.params['E21'] + poms_model.params['E22'])
if diff_i < 1000:
    E0_diff = (diff_i+1.0)*(poms_model.params['Estart'] - poms_model.params['Ereverse'])/float(30)
    priors.append(pints.NormalPrior(E0,E0_diff**2))
    priors.append(pints.NormalPrior(E0,E0_diff**2))
    E1_diff = (diff_i+1.0)*(poms_model.params['Estart'] - poms_model.params['Ereverse'])/float(30)
    priors.append(pints.NormalPrior(E1,E1_diff**2))
    priors.append(pints.NormalPrior(E1,E1_diff**2))
    E2_diff = (diff_i+1.0)*(poms_model.params['Estart'] - poms_model.params['Ereverse'])/float(30)
    priors.append(pints.NormalPrior(E2,E2_diff**2))
    priors.append(pints.NormalPrior(E2,E2_diff**2))
    if reversible:
       priors.append(pints.UniformPrior(lower_bounds[6:8],upper_bounds[6:8]))
    else:
       priors.append(pints.UniformPrior(lower_bounds[6:14],upper_bounds[6:14]))
else:
    if reversible:
       priors.append(pints.UniformPrior(lower_bounds[0:8],upper_bounds[0:8]))
    else:
       priors.append(pints.UniformPrior(lower_bounds[0:14],upper_bounds[0:14]))


# Load a forward model
pints_model = electrochemistry.PintsModelAdaptor(poms_model,names)

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(pints_model, data.time, data.current)

# Create a log-likelihood function scaled by n
log_likelihood = pints.ScaledLogLikelihood(pints.UnknownNoiseLogLikelihood(problem))

# Create a uniform prior over both the parameters and the new noise variable
prior = pints.ComposedPrior(*priors)

# Create a Bayesian log-likelihood (prior * likelihood)
score = pints.BayesianLogLikelihood(prior, log_likelihood)

# Select some boundaries
boundaries = pints.Boundaries(lower_bounds,upper_bounds)

# Perform an optimization with boundaries and hints
#if reversible:
#   x0 = [E0,E0,E1,E1,E2,E2] \
#        + [0.5*(u-l) for l,u in zip(lower_bounds[6:8],upper_bounds[6:8])]
#else:
#   x0 = [E0,E0,E1,E1,E2,E2] \
#        + [0.5*(u-l) for l,u in zip(lower_bounds[6:14],upper_bounds[6:14])]

x0 = [0.5*(u+l) for l,u in zip(lower_bounds,upper_bounds)]
sigma0 = [0.5*(h-l) for l,h in zip(lower_bounds,upper_bounds)]


print('log like at x0: ')
print(log_likelihood(x0))

print('prior at x0: ')
print(prior(x0))

print('Score at x0: ')
print(score(x0))

found_parameters, found_solution = pints.cmaes(
    score,
    boundaries,
    x0,
    sigma0,
    )

print('Found solution:          x0:' )
for k, x in enumerate(found_parameters):
    print(pints.strfloat(x) + '    ' + pints.strfloat(x0[k]))

print('Score at found_parameters: ')
print(found_solution)
#print(score(found_parameters))

if reversible:
   dir_name = 'reversible_%s'%(diff_i)
else:
   dir_name = 'quasireversible_%s'%(diff_i)

if not os.path.exists(dir_name):
   os.makedirs(dir_name)

pickle.dump( (found_parameters,found_solution), open( '%s/params_and_solution%d.p'%(dir_name,sample_i), "wb" ) )

