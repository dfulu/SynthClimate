import numpy as np
import pyclimine as pcm


#################################
# timeseries generating functions
#################################

def softplus(x): 
    return np.log(1+np.exp(x))

def increasing_angle_timeseries(n_T, period, ar_params=[], ma_params=[], sar_params=[], sma_params=[], sarma_period=1, sigma=1.):
    """For use with nonlinear cyclic and wave modes"""
    # create always increasing but chaotic time series
    s = softplus(pcm.generate.simulate_sarma(n_T, ar_params=ar_params,  ma_params=ma_params, 
                                   sar_params=sar_params, sma_params=sma_params, 
                                   period=sarma_period, sigma=sigma)).cumsum()
    # enforce average period
    s = (s/s[-1] * len(s) /  float(period))%1.
    return s

def non_linear_timeseries_fn(n_T, period, ar_params=[], ma_params=[], sar_params=[], sma_params=[], sigma=1.):
    """For use with the nonlinear but not cyclic mode type"""
    ts = pcm.generate.simulate_sarma(n_T, ar_params=ar_params,  ma_params=ma_params, 
                                   sar_params=sar_params, sma_params=sma_params, 
                                   period=period, sigma=sigma)
    
    ts = pcm.generate.sigmoid_function(ts, a=1.)
    
    return ts

def linear_time_series_constructor(ar_params, ma_params, period,
                                   sar_params, sma_params):
    return lambda n_T: pcm.generate.simulate_sarma(n_T, ar_params=ar_params,  ma_params=ma_params, 
                                   sar_params=sar_params, sma_params=sma_params, 
                                   period=int(period), sigma=1)

def nonlinear_time_series_constructor(ar_params, ma_params, period,
                                   sar_params, sma_params):
    return lambda n_T: non_linear_timeseries_fn(n_T, int(period), ar_params=ar_params,  ma_params=ma_params, 
                                   sar_params=sar_params, sma_params=sma_params, sigma=1)

def periodic_time_series_constructor(ar_params, ma_params, period,
                                   sar_params, sma_params):
    return lambda n_T: increasing_angle_timeseries(n_T, period, ar_params=ar_params,  ma_params=ma_params, 
                                sar_params=sar_params, sma_params=sma_params, sarma_period=int(period), sigma=1.)

#################################
# create mode object from summary table
#################################


def create_dataset(n_T, data_setup):
    df_setup = data_setup.summary
    data_object = pcm.generate.mixed_modes_master(
                    lats = np.arange(90,-91,-2.5),
                    lons = np.arange(0,360,3.75),
                    n_T = n_T)

    # ADD NOISE MODES
    if 'noise' in df_setup.type.values:
        var = df_setup.loc[df_setup.type=='noise', 'variance'][0]
        
        data_object.initiate_noise_modes()
        data_object.noise_modes.add_noise_modes()
        data_object.noise_modes.implement_timeseries_functions(
                    vr=var)
    
    # ADD LINEAR MODES
    if  ('linear_sparse' in df_setup.type.values) or \
        ('linear_dense' in df_setup.type.values):
        data_object.initiate_linear_modes()
        mode_setup_df =  df_setup.loc[
                (df_setup.type=='linear_sparse')|(df_setup.type=='linear_dense')
                ].sort_values('type')
        
        n_dense  = sum(df_setup.type=='linear_dense')
        n_sparse = sum(df_setup.type=='linear_sparse')
        
        fns = [linear_time_series_constructor(*p) for p in mode_setup_df.loc[:, 'timeseries_info'].values]
        vrs = mode_setup_df.loc[:, 'variance'].values
        
        data_object.linear_modes.add_linear_modes(n_modes=n_dense, kernel_ells=[30], 
                    data_distribution_draw_function=pcm.generate.normal_dist_draw_func)
        data_object.linear_modes.add_linear_modes(n_modes=n_sparse, kernel_ells=[30], 
                    data_distribution_draw_function=pcm.generate.sparse_dist_draw_func)
            
        data_object.linear_modes.implement_timeseries_functions(
                    fns = fns, 
                    vrs = vrs)
        
    # NON-LINEAR MODES
    if  'non_linear' in df_setup.type.values:
        data_object.initiate_non_linear_modes()
        mode_setup_df =  df_setup.loc[df_setup.type=='non_linear']
        n_nonlin = len(mode_setup_df)
        
        fns = [nonlinear_time_series_constructor(*p) for p in mode_setup_df.loc[:, 'timeseries_info'].values]
        vrs = mode_setup_df.loc[:, 'variance'].values
            
        data_object.non_linear_modes.add_non_linear_modes(n_nonlin)
        data_object.non_linear_modes.implement_timeseries_functions(
                    fns=fns, 
                    vrs = vrs)
        
    # NON-LINEAR CYCLIC MODES
    if  'non_linear_cyclic' in df_setup.type.values:
        data_object.initiate_non_linear_cyclic_modes()
        mode_setup_df =  df_setup.loc[df_setup.type=='non_linear_cyclic']
        n_nonlincyc = len(mode_setup_df)
        
        fns = [periodic_time_series_constructor(*p) for p in mode_setup_df.loc[:, 'timeseries_info'].values]
        vrs = mode_setup_df.loc[:, 'variance'].values
            
        data_object.non_linear_cyclic_modes.add_non_linear_modes(n_nonlincyc)
        data_object.non_linear_cyclic_modes.implement_timeseries_functions(
                    fns=fns, 
                    vrs = vrs)
        
    # WAVE MODES
    if  'moving_wave' in df_setup.type.values:
        data_object.initiate_wave_modes()
        mode_setup_df =  df_setup.loc[df_setup.type=='moving_wave']
        n_wave = len(mode_setup_df)
        
        fns = [periodic_time_series_constructor(*p) for p in mode_setup_df.loc[:, 'timeseries_info'].values]
        vrs = mode_setup_df.loc[:, 'variance'].values
        path_fns = [pcm.generate.default_phase_to_lat_lon_path]
         
        data_object.wave_modes.add_traversing_modes(n_wave)
        data_object.wave_modes.implement_timeseries_functions(
                    fns=fns,
                    phase_to_lat_lon_fns=path_fns,
                    vrs = vrs)
        
    return data_object

    
if __name__=='__main__':
    from random_draw import pseudoclim
    n_T=100
    seed=3412892
    data_setup = pseudoclim(seed)
    data_object = create_dataset(n_T, data_setup)
    print(data_object.summary)
    # data_object.animate_data(flatten=False, save_path='test0', figsize=(6,8), share_colorbar=False)