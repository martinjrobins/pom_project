import pints
import pints.electrochemistry
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
import copy


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


model_base = pints.electrochemistry.POMModel(dim_params)
data = pints.electrochemistry.ECTimeData(filename,model_base,ignore_begin_samples=5,ignore_end_samples=0)

def get_prior(i,reversible):
    model = model_base
    prior = pints.Prior()
    e0_buffer = 0.1*(model.params['Estart'] - model.params['Ereverse'])
    E0 = 0.5*(model.params['E01'] + model.params['E02'])
    E0_diff = (i+1.0)*(model.params['Estart'] - model.params['Ereverse'])/float(n)
    print 'E0_diff (dim) = ',E0_diff*model.E0
    prior.add_parameter('E01',pints.Normal(E0,(E0_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    prior.add_parameter('E02',pints.Normal(E0,(E0_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    E1 = 0.5*(model.params['E11'] + model.params['E12'])
    E1_diff = E0_diff
    prior.add_parameter('E11',pints.Normal(E1,(E1_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    prior.add_parameter('E12',pints.Normal(E1,(E1_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    E2 = 0.5*(model.params['E21'] + model.params['E22'])
    E2_diff = E0_diff
    prior.add_parameter('E21',pints.Normal(E2,(E2_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    prior.add_parameter('E22',pints.Normal(E2,(E2_diff)**2),
            model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
    prior.add_parameter('Cdl',pints.Uniform(),0,20)
    if reversible:
        prior.add_parameter('k01',pints.Uniform(),0,10000)
        prior.add_parameter('k02',pints.Uniform(),0,10000)
        prior.add_parameter('k11',pints.Uniform(),0,10000)
        prior.add_parameter('k12',pints.Uniform(),0,10000)
        prior.add_parameter('k21',pints.Uniform(),0,10000)
        prior.add_parameter('k22',pints.Uniform(),0,10000)
    else:
        names = ['k01','k02','k11','k12','k21','k22']
        model.set_params_from_vector([1e10 for name in names],names)

    prior.add_parameter('Ru',pints.Uniform(),0,0.1)

    omega_est = model.params['omega']
    prior.add_parameter('omega',pints.Normal(omega_est,(omega_est*0.01)**2),0.9*omega_est,1.1*omega_est)
    prior.add_parameter('phase',pints.Normal(model.params['phase'],(pi/10.0)**2),-pi/5,pi/5)
    prior.add_parameter('gamma',pints.Uniform(),0.1,10.0)

    return prior


priors = []
run_names = []
n = 30
for i in range(n):
    prior = get_prior(i,False)
    priors.append(prior)
    run_names.append('quasi_reversible_%d'%i)

for i in range(n):
    prior = get_prior(i,True)
    priors.append(prior)
    run_names.append('reversible_%d'%i)

for dir_name_base,prior_base in zip(run_names,priors):
    if not os.path.exists(dir_name_base):
        os.makedirs(dir_name_base)

    data_log_likelihoods = np.zeros(30,'float')
    for i in range(30):
        dir_name = '%s/sample_%d'%(dir_name_base,i)
        model = copy.copy(model_base)
        prior = copy.copy(prior_base)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        f = open('%s/out'%dir_name,'w')
        sys.stdout = f

        print 'std devations are:'
        for name in ['E01','E02','E11','E12','E21','E22']:
            print '\t',name,': ',math.sqrt(prior.data[name][0].variance)


        print 'before cmaes, parameters are:'
        names = prior.get_parameter_names()
        v = np.zeros(len(names))
        for i in range(len(names)):
            v[i] = model.params[names[i]]
            print names[i],': ',model.params[names[i]]

        print 'param loglikelihood is currently',prior.log_likelihood(v,names)

        pints.fit_model_with_cmaes(data,model,prior)


        print 'after cmaes, parameters are:'
        names = prior.get_parameter_names()
        for name in prior.get_parameter_names():
            print name,': ',model.params[name]

        print 'after cmaes, dim parameters are:'

        names = prior.get_parameter_names()
        for name in prior.get_parameter_names():
            print name,': ',model.dim_params[name]

        I,t = model.simulate(use_times=data.time)

        data_log_likelihoods[i] = data.log_likelihood(I,1)

        print 'data log_likelihood = ',data_log_likelihoods[i]

        pickle.dump( (model,prior,dir_name), open( '%s/model_prior_dir_name.p'%dir_name, "wb" ) )

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
        f.savefig('%s/fit_pom_harmonics.pdf'%dir_name)
        fabs.tight_layout()
        fabs.savefig('%s/fit_pom_harmonics_abs.pdf'%dir_name)


        plt.figure()
        plt.plot(t,I,alpha=0.5)
        plt.plot(data.time,data.current,alpha=0.5)
        plt.savefig('%s/fit_pom.pdf'%dir_name)
        plt.close()

    pickle.dump( (data_log_likelihoods), open( '%s/data_log_likelihoods.p'%dir_name_base, "wb" ) )



