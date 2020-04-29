import numpy as np
import pyclimine as pcm
import os
import sys
import gc
import matplotlib.pyplot as plt
from datetime import datetime
import psutil

from random_draw import pseudoclim
from fulfil_draw import create_dataset

# set plot backend
from matplotlib import use as mpl_use
mpl_use('Agg')

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

def instantiate_model(test, n_modes):
    if modname=='sfa':
        return test.instantiate_model(n_components=n_modes, pca_components=n_modes)
    else:
        return test.instantiate_model(n_components=n_modes)

#########################
#  Test Setup
#########################
    
# output file


directory_base  = '/exports/csce/datastore/geos/users/s1205782/evaluation/{}'.format(modname) + \
                  '/mixed_modes/trial_{:03}'

# Key test numbers
n_T = 14400 # length of time series
n_trials = 60 # number of trials to run
# Derived
seeds = 356723 + np.arange(n_trials)

# Plotting and data dump options
create_match_animation=False
dump_data_plots = True if modname =='pca' else False
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
# RUN
#################################

for i in range(n_trials):
    
    directory = directory_base.format(i)
    
    if os.path.exists(directory):
        print('WARNING - {} already exists - SKIPPING')
        continue
    
    extprint('- directory path')
    print(directory)
    
    extprint('- creating and populating new data object')
    mode_draw = pseudoclim(seeds[i])
    data_object = create_dataset(n_T, mode_draw)
    n_modes = int(mode_draw.n_modes + 2)
    
    extprint('- set up done')
    #########################
    # RUN TEST
    #########################    

    print('---------------RUNNING TEST---------------')
    # instantiate test and process data
    extprint('- initiating test')
    test = test_object(data_object, directory, overwrite=False)
    mode_draw.summary.to_csv(os.path.join(directory, 'setup.csv'))
    extprint('- processing data')
    test.process_data()
    # instantiate and train model on data
    extprint('- initiating model')
    instantiate_model(test, n_modes)
    extprint('- training model')
    test.train_model()
    # run the metric test on the model
    extprint('- testing model')
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