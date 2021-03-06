import numpy as np
import pyclimine as pcm
import os
import sys
import gc
import matplotlib.pyplot as plt
from datetime import datetime
import psutil

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
                    '/time_sensitivity/trial_{:03}_timelen_{:02}'

# Key test numbers
n_modes=12 # number of model modes
timelens = [y*12 for y in [20, 40, 100, 200, 400, 800, 2000]] # length of datasets
repeats = 6
# Derived
timelens = np.repeat(timelens, repeats=repeats, axis=0)
trial_ns = np.tile(np.arange(repeats), len(timelens))
seeds = 992737 + trial_ns

# Plotting and data dump options
create_match_animation=False
dump_data_plots= False
figsize=(8,33)

#################################
# extended print function
#################################

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 10**9

def extprint(x):
    print('{} (time: {}) (memory : {:.2f} GB)'.format(x, str(datetime.now())[:-7], memory_usage()))
    return

#################################
# Data setup
#################################

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


for i, (trial_n, n_T, seed) in enumerate(zip(trial_ns, timelens, seeds)):
    
    print('---------------GENERATING DATA------------')
    
    directory = directory_base.format(trial_n, n_T)
    
    extprint('- starting')
    print(directory)
    
    if os.path.exists(directory):
        print('WARNING - {} already exists - SKIPPING'.format(directory))
        continue
    
    extprint('- creating new data object')
    
    data_object = pcm.generate.mixed_modes_master(
                    lats = np.arange(90,-91,-2.5),
                    lons = np.arange(0,360,3.75),
                    n_T = n_T)
    
    # SET SEED
    np.random.seed(seed)
    
    extprint('- populating data object')
    # LINEAR
    data_object.initiate_linear_modes()
    data_object.linear_modes.add_linear_modes(n_modes=8, kernel_ells=[30], 
                        data_distribution_draw_function=pcm.generate.normal_dist_draw_func)
    data_object.linear_modes.implement_timeseries_functions(fns = [tf0, tf1, tf2, tf3]*2, 
                                                            vrs = [1])

    # NOISE
    data_object.initiate_noise_modes()
    data_object.noise_modes.add_noise_modes()
    data_object.noise_modes.implement_timeseries_functions(vr=1)
    
    extprint('- set up done')

    #########################
    # RUN TEST
    #########################    

    print('---------------RUNNING TEST---------------')
    # instantiate test and process data
    extprint('- initiating test')
    test = test_object(data_object, directory, overwrite=False)
    extprint('- processing data')
    test.process_data()
    # instantiate and train model on data
    extprint('- initiating model')
    instantiate_model(test)
    extprint('- training model')
    test.train_model()
    # run the metric test on the model
    extprint('- running test_model')
    test.test_model()
    extprint('- saving test summary')
    test.save_test_summary()
    # create and save the base plots
    extprint('- generating and saving core plots')
    test.save_model_plots(extra=False)
    # create and save the model specific plots
    extprint('- generating and saving other plots')
    test.save_model_plots(extra=True)
    
    # generate and save the model and summary
    if dump_data_plots:
        extprint('- dumping generated data plots')
        dirpath = os.path.join(directory, 'generated_data_plots')
        data_object.dump_plots(directory=dirpath, non_lin_gif=True)
        
    if create_match_animation:
        extprint('- creating match animation')
        fpath = os.path.join(directory, 'animated_matches')
        test.animate_matches(save_path=fpath, match_indexes=[], figsize=figsize,
                                 t_ind_start=0, t_ind_stop=100)
        plt.close('all')
        
    # print results
    print('\n'*2)
    print('overall metric : ', test.overall_metric)
    print('-'*8+'DONE'+'-'*8)
    
    # delete things from memory
    extprint('- clearing up')
    plt.close('all')
    del data_object
    del test
    gc.collect()
    extprint('- clear up finished')
    
