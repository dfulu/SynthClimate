import numpy as np
import pyclimine as pcm
import os
import sys

import matplotlib
matplotlib.use('Agg')

#########################
#  Input options
#########################

# model short name must be put in as initials
modname = sys.argv[1]

test_objects = {'pca': pcm.evaluate.pca_test_object,
               'sfa': pcm.evaluate.sfa_test_object,
               'base': pcm.evaluate.baseline_test_object,
               'ann': pcm.evaluate.autoencoder_test_object,
               'kpca': pcm.evaluate.kpca_test_object,
               'ica': pcm.evaluate.ica_test_object,
               'fa': pcm.evaluate.fa_test_object,
               'dmd': pcm.evaluate.dmd_test_object,
               'nmf': pcm.evaluate.nmf_test_object,}

test_object = test_objects[modname]

def instantiate_model(test):
    if modname=='sfa':
        return test.instantiate_model(n_components=n_modes, pca_components=n_modes)
    else:
        return test.instantiate_model(n_components=n_modes)



#########################
#  Test Setup
#########################
    
# output file
directory_base  =   '/exports/csce/datastore/geos/users/s1205782/evaluation/{}'.format(modname) + \
                    '/variance_sensitivity/trial_{:03}_leading_var_{:02}'

# Key test numbers
n_modes=12 # number of model modes
n_T = 14400 # length of time series
leading_vars = [2,4,8,16,32]
repeats = 6
# Derived
mode_vrs = np.array([[i]+[1]*7 for i in leading_vars])
mode_vrs = np.repeat(mode_vrs, repeats=repeats, axis=0)
trial_ns = np.tile(np.arange(repeats), len(leading_vars))
seeds = 82737 + trial_ns

# Plotting and data dump options
create_match_animation=False
dump_data_plots= False





figsize=(8,33)

data_object = pcm.generate.mixed_modes_master(
                    lats = np.arange(90,-91,-2.5),
                    lons = np.arange(0,360,3.75),
                    n_T = n_T)

# timeseries generating functions
tf0 = pcm.generate.default_timeseries_fn

tf1 = lambda n_T: pcm.generate.simulate_sarma(n_T, 
                            ar_params=[0.1, 0.2, 0.1], ma_params=[-0.3], 
                            sar_params=[0.33], sma_params=[0.1], 
                            period=36, sigma=1.)

tf2 = lambda n_T: pcm.generate.simulate_sarma(n_T, 
                            ar_params=[-0.3, 0., 0.1], ma_params=[], 
                            sar_params=[0.5], sma_params=[0.1], 
                            period=19, sigma=1.)

tf3 = lambda n_T: pcm.generate.simulate_sarma(n_T, 
                            ar_params=[-0.1, 0., 0.1], ma_params=[0.77], 
                            sar_params=[], sma_params=[], 
                            period=1, sigma=1.)

print('GENERATING DATA')
for i, (trial_n, vrs, seed) in enumerate(zip(trial_ns, mode_vrs, seeds)):
    
    leading_vr = vrs[0]
    directory = directory_base.format(trial_n, leading_vr)
    
    # SET SEED
    np.random.seed(seed)
    
    # LINEAR
    data_object.initiate_linear_modes()
    data_object.linear_modes.add_linear_modes(n_modes=8, kernel_ells=[30], 
                        data_distribution_draw_function=pcm.generate.normal_dist_draw_func)
    data_object.linear_modes.implement_timeseries_functions(fns = [tf0, tf1, tf2, tf3]*2, 
                                                            vrs = vrs)

    # NOISE
    data_object.initiate_noise_modes()
    data_object.noise_modes.add_noise_modes()
    data_object.noise_modes.implement_timeseries_functions(vr=1)
    
    #########################
    # RUN TEST
    #########################    

    print('RUNNING TEST')
    # instantiate test and process data
    print('- initiating test')
    test = test_object(data_object, directory, overwrite=False)
    print('- processing data')
    test.process_data()
    # instantiate and train model on data
    print('- initiating model')
    instantiate_model(test)
    print('- training model')
    test.train_model()
    # run the metric test on the model
    print('- running test_model')
    test.test_model()
    print('- saving test summary')
    test.save_test_summary()
    # create and save the base plots
    print('- generating and saving core plots')
    test.save_model_plots(extra=False)
    # create and save the model specific plots
    print('- generating and saving other plots')
    test.save_model_plots(extra=True)
    
    # generate and save the model and summary
    if dump_data_plots:
        print('- dumping generated data plots')
        fpath = os.path.join(directory, 'generated_data_plots')
        data_object.dump_plots(directory='', non_lin_gif=False)
    if create_match_animation:
        print('- creating match animation')
        fname = os.path.join(directory, 'animated_matches')
        test.animate_matches(save_path='', match_indexes=[], figsize=figsize,
                                 t_ind_start=0, t_ind_stop=100)
    print('\n'*8)
    print(test.test_summary)
    print('overall metric : ', test.overall_metric)
    print('-'*8+'DONE'+'-'*8)
    
