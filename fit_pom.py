import hobo
import hobo.electrochemistry
import pickle

import matplotlib as mpl
#mpl.use('Agg')
import pylab as plt


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


model = hobo.electrochemistry.POMModel(dim_params)

data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

# specify bounds for parameters
prior = hobo.Prior()
e0_buffer = 0.1*(model.params['Estart'] - model.params['Ereverse'])
prior.add_parameter('E01',hobo.Normal(0.35,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E02',hobo.Normal(0.35,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E11',hobo.Normal(0.227,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E12',hobo.Normal(0.227,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E21',hobo.Normal(0.0,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('E22',hobo.Normal(0.0,0.05),
        model.params['Ereverse']+e0_buffer,model.params['Estart']-e0_buffer)
prior.add_parameter('k01',hobo.Uniform(),0,10000)
prior.add_parameter('k02',hobo.Uniform(),0,10000)
prior.add_parameter('k11',hobo.Uniform(),0,10000)
prior.add_parameter('k12',hobo.Uniform(),0,10000)
prior.add_parameter('k21',hobo.Uniform(),0,10000)
prior.add_parameter('k21',hobo.Uniform(),0,10000)

print 'before cmaes, parameters are:'
names = prior.get_parameter_names()
for name in prior.get_parameter_names():
    print name,': ',model.params[name]

#model.set_params_from_vector([0.00312014718956,2.04189332425,7.274953392],['Cdl','k0','E0'])
hobo.fit_model_with_cmaes(data,model,prior)

print 'after cmaes, parameters are:'
names = prior.get_parameter_names()
for name in prior.get_parameter_names():
    print name,': ',model.params[name]


I,t = model.simulate(use_times=data.time)
plt.figure()
plt.plot(t,I)
plt.plot(data.time,data.current)
plt.savefig('fit_pom.pdf')
plt.close()


