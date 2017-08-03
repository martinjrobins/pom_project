import pints
from pints import electrochemistry
import pickle
import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import pylab as plt
import numpy.fft as fft
from math import pi


filename = 'POMGCL_6020104_1.0M_SFR_d16.txt'

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
    'k01': 7300,
    'k02': 7300,
    'k11': 1e4,
    'k12': 1e4,
    'k21': 2500,
    'k22': 2500
    }


model = electrochemistry.POMModel(dim_params)
data = electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

I,t = model.simulate(use_times=data.time)

print 'data loglikelihood is currently',data.log_likelihood(I,0.5)

# specify bounds for parameters
prior = pints.Prior()
e0_buffer = 0.1*(model.params['Estart'] - model.params['Ereverse'])
E0 = 0.5*(model.params['E01'] + model.params['E02'])
E0_diff = 1
prior.add_parameter('E01',pints.Normal(E0,(2*E0_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E02',pints.Normal(E0,(2*E0_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
E1 = 0.5*(model.params['E11'] + model.params['E12'])
E1_diff = 1
prior.add_parameter('E11',pints.Normal(E1,(2*E1_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E12',pints.Normal(E1,(2*E1_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
E2 = 0.5*(model.params['E21'] + model.params['E22'])
E2_diff = 1
prior.add_parameter('E21',pints.Normal(E2,(2*E2_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E22',pints.Normal(E2,(2*E2_diff)**2),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('k01',pints.Uniform(),0,10000)
prior.add_parameter('k02',pints.Uniform(),0,10000)
prior.add_parameter('k11',pints.Uniform(),0,10000)
prior.add_parameter('k12',pints.Uniform(),0,10000)
prior.add_parameter('k21',pints.Uniform(),0,10000)
prior.add_parameter('k22',pints.Uniform(),0,10000)
prior.add_parameter('gamma',pints.Uniform(),0.1,10.0)

print 'before cmaes, parameters are:'
names = prior.get_parameter_names()
v = np.zeros(len(names))
for i in range(len(names)):
    v[i] = model.params[names[i]]
    print names[i],': ',model.params[names[i]]

print 'param loglikelihood is currently',prior.log_likelihood(v,names)

#

v = [351.799992423,
  4374.66433409,
  6948.72148094,
  -0.0171315128634,
  -0.587492063523,
  5965.6272421,
  8.53884480798,
  9.13625737772,
  13.6671653148,
  2184.96104188,
  14.0741162244,
  885.199106042,
  0.808733615878]
names = ['k01',
         'k02',
        'k21',
        'E21',
        'E22',
        'k12',
        'E11',
        'E12',
        'E02',
        'k11',
        'E01',
        'k22',
        'gamma']
model.set_params_from_vector(v,names)

#pints.fit_model_with_cmaes(data,model,prior,IPOP=[10,20,40,80,160])
#pints.fit_model_with_cmaes(data,model,prior)

print 'after cmaes, parameters are:'
names = prior.get_parameter_names()
for name in prior.get_parameter_names():
    print name,': ',model.params[name]

print 'after cmaes, dim parameters are:'

names = prior.get_parameter_names()
for name in prior.get_parameter_names():
    print name,': ',model.dim_params[name]



I,t = model.simulate(use_times=data.time)
t = np.array(t)
F = (fft.fft(I),fft.fft(data.current))
dt = t[1]-t[0]
signal_f = model.params['omega']/(2*pi)
ia = np.array([1,2,3,4,5,6,7,8])
hs = ia*signal_f
lowcut = hs - signal_f/4
highcut = hs + signal_f/4
f,axs = plt.subplots(4,2,figsize = (10,13))
fabs,axs_abs = plt.subplots(4,2,figsize = (10,13))
colors = ['blue','red']

weights = np.zeros(len(I))
nyq = 0.5/dt
mod = 1
T_0 = model.T0
I_0 = model.I0
for i in range(len(ia)):
    print 'doing subplot',int(i/2),',',i%2
    cfreq = int(len(F[0]) * hs[i]/nyq/2)
    band_low = int(lowcut[i]/nyq*len(F[0])/2)
    band_high = int(highcut[i]/nyq*len(F[0])/2)
    weights[:] = 0.0
    weights[band_low:band_high] = 2.0
    ax = axs[int(i/2),i%2]
    ax_abs = axs_abs[int(i/2),i%2]
    for color,FI in zip(colors,F):
        Ih = FI*weights
        Ih = np.concatenate((Ih[cfreq:],Ih[0:cfreq]))
        Ih = fft.ifft(Ih)

        if len(I)%2 == 0:
            ax.plot(t[0::mod]*T_0,np.real(Ih[0::mod])*I_0*1e6,c=color,ls='-')
            ax.plot(t[0::mod]*T_0,np.imag(Ih[0::mod])*I_0*1e6,c=color,ls='--')
            ax_abs.plot(t[0::mod]*T_0,np.abs(Ih[0::mod])*I_0*1e6,c=color,ls='-')
        else:
            ax.plot(T_0*t[0::mod],np.real(Ih[0::mod])*I_0*1e6,c=color,ls='-')
            ax.plot(T_0*t[0::mod],np.imag(Ih[0::mod])*I_0*1e6,c=color,ls='--')
            ax_abs.plot(T_0*t[0::mod],np.abs(Ih[0::mod])*I_0*1e6,c=color,ls='-')

    if i%2==1:
        ax.set_ylabel(r'$|I_{%dth}| \, (\mu A)$'%ia[i])
       #ax.set_yticks([])
    else:
        ax.set_ylabel(r'$I_{%dth} \, (\mu A)$'%ia[i])
    if int(i/2)<4:
        ax.set_xticks([])
    else:
        ax.set_xlabel('$t \, (s)$')
                                                                                                        #axs[0].legend()
f.tight_layout()
f.savefig('fit_pom_harmonics.pdf')
fabs.tight_layout()
fabs.savefig('fit_pom_harmonics_abs.pdf')


plt.figure()
plt.plot(t,I,alpha=0.5)
plt.plot(data.time,data.current,alpha=0.5)
plt.savefig('fit_pom.pdf')
plt.close()



