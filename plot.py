#!/usr/bin/env python2
import matplotlib.pyplot as plt
import glob
import numpy as np
import pickle
from math import pi
import re
import sys
sys.path.insert(0, '../pints')
sys.path.insert(0, '../pints/problems/electrochemistry')
sys.path.insert(0, '../build')
import pints
import electrochemistry


def plot_harmonic_1_6_full(poms_model, Is, t, output_filename, mod=1):
    print 'plot_harmonic_1_6_full....', output_filename
    colors = ['blue', 'red', 'green']

    E_0, T_0, L_0, I_0 = poms_model._calculate_characteristic_values()
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)
        ) = plt.subplots(3, 2, figsize=(10, 8.4))
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    dt = t[1]-t[0]
    freqs = np.fft.rfftfreq(len(Is[0]), d=dt)
    omega = poms_model.params['omega']
    signal_f = omega/(2*pi)
    ia = np.array([1, 2, 3, 4, 5, 6])
    hs = ia*signal_f
    lowcut = hs - signal_f/4
    highcut = hs + signal_f/4

    print 'plot_harmonic_1_6_full....calculating harmonics'
    for i in range(len(ia)):
        for n, I in zip(range(len(Is)), Is):
            ax = axs[i]
            #F = fft.rfft(I)*ws[i]
            F = np.fft.fft(I)
            if n == 0:
                label = 'experiment'
            else:
                label = 'simulation'
            signal_f = poms_model.params['omega']/(2*pi)
            nyq = 0.5 / (t[1]-t[0])
            hs = ia[i]*signal_f
            band_low = int((hs-0.5*signal_f)/nyq*len(F)/2)
            band_high = int((hs+0.5*signal_f)/nyq*len(F)/2)
            cfreq = int(len(F) * hs/nyq/2)
            pos_index_low = band_low
            pos_index_high = band_high
            neg_index_low = len(F) - band_high
            neg_index_high = len(F) - band_low
            F_filt = np.zeros(len(F), dtype=complex)
            F_filt[pos_index_low:pos_index_high] = F[pos_index_low:pos_index_high]
            #F_filt[0:0.5*cfreq+1] = 2*F[pos_index_low+0.5*cfreq:pos_index_high]
            #F_filt[len(F)-0.5*cfreq+1:] = 2*F[pos_index_low:pos_index_high-0.5*cfreq]

            if cfreq > 0:
                F = np.concatenate((F_filt[cfreq:], np.zeros(
                    len(F)-2, dtype=complex), F_filt[0:cfreq]))
            else:
                F = np.concatenate((F_filt, np.conj(F_filt[-2:0:-1])))

            #wsia[pos_index_low:pos_index_high] = 2.0
            #wsia[neg_index_low:neg_index_high] = 1.0
            #wsia[(cfreq-band_high):(cfreq-band_low)] = 1.0
            #F = F*np.concatenate((ws[i]*2,np.zeros(len(F)-len(ws[i]),dtype=complex)))
            #F = F*wsia

            F = np.fft.ifft(F_filt)
            F = 2*np.abs(F)

            if len(I) % 2 == 0:
                ax.plot(t[0::mod]*T_0, F[0::mod]*I_0*1e6,
                        label=label, c=colors[n], ls='-')
            else:
                ax.plot(T_0*t[0::mod], F[0::mod]*I_0*1e6,
                        label=label, c=colors[n], ls='-')

        if ia[i] >= 4:
            ax.set_ylabel(r'$|I_{%dth}| \, (\mu A)$' % ia[i])
        elif ia[i] == 1:
            ax.set_ylabel(r'$|I_{%dst}| \, (\mu A)$' % ia[i])
        elif ia[i] == 2:
            ax.set_ylabel(r'$|I_{%dnd}| \, (\mu A)$' % ia[i])
        elif ia[i] == 3:
            ax.set_ylabel(r'$|I_{%drd}| \, (\mu A)$' % ia[i])

        if i < 4:
            ax.set_xticks([])
        else:
            ax.set_xlabel('$t \, (s)$')
    # axs[0].legend()

    plt.tight_layout()
    f.savefig(output_filename+'.pdf')
    plt.close(f)


filename = '../POMGCL_6020104_1.0M_SFR_d16.txt'
plot_sim = True

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
    'Cdl': 0.000010,
    'Gamma': 0.7*53.0e-12,
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
data = electrochemistry.ECTimeData(
    filename, poms_model, ignore_begin_samples=5, ignore_end_samples=0)

names = {'reversible':
         ['E01',
          'E02',
          'E11',
          'E12',
          'E21',
          'E22',
          'alpha1',
          'alpha2',
          'Ru',
          'Cdl',
          'gamma'],
         'quasireversible':
         ['E01',
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
          'alpha1',
          'alpha2',
          'Ru',
          'Cdl',
          'gamma']}


for reaction_type in ['reversible', 'quasireversible']:
    pints_model = electrochemistry.PintsModelAdaptor(
        poms_model, names[reaction_type])

    if reaction_type == 'quasireversible':
        heuristic_params = [0.368, 0.338, 0.227, 0.227, 0.011, -
                            0.016, 7300.0, 7300.0, 10000.0, 10000.0, 2500.0, 2500.0, 0.5, 0.5, 50, 0.000010, 1.0]
        heuristic_nondim_params = [poms_model.non_dimensionalise(x, name)
                                   for x, name in zip(heuristic_params, names[reaction_type])]
        print heuristic_nondim_params
        current = pints_model.simulate(
            np.array(heuristic_nondim_params), times=data.time)
        print np.array(current)
        f = plt.figure()
        plt.plot(data.time, current, label='sim')
        plt.plot(data.time, data.current, label='exp')
        plt.legend()
        plt.savefig('heuristic_vis.pdf')
        plt.close(f)
        plot_harmonic_1_6_full(poms_model, [
                               current, data.current], data.time, 'heuristic_visHarmonics.pdf', mod=1)

    dir_names = sorted(glob.glob(reaction_type+'_*'))
    dir_names = [reaction_type+'_2']
    n_samples = len(glob.glob(dir_names[0]+'/params_and_solution*.p'))
    #n_samples = 20
    score = np.zeros(len(dir_names)*n_samples)
    stddev = np.zeros(len(dir_names)*n_samples)
    num_good_fits = np.zeros(len(dir_names))
    stddev_num_good_fits = np.zeros(len(dir_names))
    for i, dir_name in enumerate(dir_names):
        diff_i = re.match('.*?([0-9]+)$', dir_name).group(1)
        print 'dir_name =', dir_name
        print 'diff_i =', diff_i
        min_score = 1000
        for sample in range(n_samples):
            print sample
            sample_filename = dir_name+'/params_and_solution%d.p' % (sample+1)
            sample_params, sample_score = pickle.load(open(sample_filename))
            #print 'Non-dimensional params:', sample_params

            print 'Dimensional params:'
            dim_val_file = open(dir_name+'/dim_params%d.txt' % (sample+1), 'w')
            for val, name in zip(sample_params, names[reaction_type]):
                print name, '=', poms_model.dimensionalise(val, name), val
                dim_val_file.write(
                    str(name)+" = "+str(poms_model.dimensionalise(val, name)))
            dim_val_file.close()

            score[i*n_samples+sample] = sample_score
            min_score = min(min_score, sample_score)
            if int(diff_i) == 1000:
                stddev[i*n_samples+sample] = (len(dir_names)+1.0)/30.0
            else:
                stddev[i*n_samples+sample] = (int(diff_i)+1.0)/30.0
                print 'stddev =', stddev[i*n_samples+sample]
        num_good_fits[i] = 0
        if int(diff_i) == 1000:
            stddev_num_good_fits[i] = (len(dir_names)+1.0)/30.0
        else:
            stddev_num_good_fits[i] = (int(diff_i)+1.0)/30.0
        for sample in range(n_samples):
            sample_filename = dir_name+'/params_and_solution%d.p' % (sample+1)
            sample_params, sample_score = pickle.load(open(sample_filename))
            if (sample_score < 1.01*min_score):
                num_good_fits[i] = num_good_fits[i] + 1.0

            if plot_sim:
                current = pints_model.simulate(
                    sample_params[:-1], times=data.time)
                f = plt.figure()
                plt.plot(data.time, current, label='sim')
                plt.plot(data.time, data.current, label='exp')
                plt.legend()
                plt.savefig(dir_name+'/vis%d_%f.pdf' %
                            (sample+1, sample_score))
                plt.close(f)

                plot_harmonic_1_6_full(poms_model, [
                                       current, data.current], data.time, dir_name+'/visHarmonics%d_%f.pdf' % (sample+1, sample_score), mod=1)

    fig, ax1 = plt.subplots()
    ax1.plot(stddev, score, '.')
    ax1.set_ylabel(r'$\mathcal{F}(\mathbf{p})$')
    ax1.set_xlabel(r'$\sigma_0$')
    ax2 = ax1.twinx()
    if reaction_type == 'reversible':
        stddev_num_good_fits, num_good_fits = [
            [x for x, _ in sorted(
                zip(stddev_num_good_fits, num_good_fits)) if x < 0.21],
            [x for y, x in sorted(zip(stddev_num_good_fits, num_good_fits)) if y < 0.21]]
    else:
        stddev_num_good_fits, num_good_fits = [
            [x for x, _ in sorted(
                zip(stddev_num_good_fits, num_good_fits)) if x < 0.25],
            [x for y, x in sorted(zip(stddev_num_good_fits, num_good_fits)) if y < 0.25]]
    ax2.plot(stddev_num_good_fits, num_good_fits, 'rx-')
    ax2.set_ylabel('number of good fits', color='r')
    ax2.tick_params('y', colors='r')
    plt.title(reaction_type)
    plt.savefig(reaction_type+'.pdf')
