import numpy as np
import pandas as pd

class pseudoclim(object):
    ####################
    # global proportions
    ####################
    
    # ----- MODE TYPE ------------
    p_L = 0.45 # p(linear pattern)
    p_N = 0.45 # p(nonlinear pattern)
    p_W = 0.1  # p(moving wave pattern)
    assert((p_L+p_N+p_W)==1), 'pattern proportions must sum to one'

    p_LS = 0.5 # p(sparse pattern | linear pattern)
    p_LD = 0.5 # p(dense pattern | linear pattern)
    assert((p_LS+p_LD)==1), 'linear pattern split proportions must sum to one'

    p_NC = 0.35 # p(cyclic pattern | nonlinear pattern)
    p_NN = 0.65 # p(not cyclic pattern | nonlinear pattern)
    assert((p_NN+p_NC)==1), 'nonlinear pattern split proportions must sum to one'
    
    mode_type_probabilities = {
        'linear_dense': p_L*p_LD, 
        'linear_sparse': p_L*p_LS, 
        'non_linear': p_N*p_NN, 
        'non_linear_cyclic': p_N*p_NC, 
        'moving_wave': p_W,
    }
    
    # ----- TIME SERIES TYPE ------------
    p_L_seasonal = 0.3 # p(seasonal time series | linear pattern)
    p_L_nonseasonal = 0.7 # p(not seasonal time series | linear pattern)
    assert((p_L_seasonal+p_L_nonseasonal)==1), 'linear seasonal split proportions must sum to one'
    
    # ----- TIME SERIES TYPE ------------
    p_NL_seasonal = 0.3 # p(seasonal time series | plain non linear pattern)
    p_NL_nonseasonal = 0.7 # p(not seasonal time series | plain non linear pattern)
    assert((p_NL_seasonal+p_NL_nonseasonal)==1), 'nonlinear seasonal split proportions must sum to one'
    
    # ----- TIME SERIES TYPE ------------
    p_P_seasonal = 0.3 # p(seasonal time series | periodic pattern)
    p_P_nonseasonal = 0.7 # p(not seasonal time series | periodic pattern)
    assert((p_P_seasonal+p_P_nonseasonal)==1), 'periodic seasonal split proportions must sum to one'
    
    # ------ ARMA coefficient --------------
    max_ar=0.8 # used for local ar coefs
    min_ar=0.1 # used for local ar coefs
    max_ma=0.1 # used for local ar coefs
    min_ma=-0.2 # used for local ar coefs
    max_sar=0.8 # used for seasonal ar coefs
    min_sar=0.1 # used for seasonal ar coefs
    max_sma=0.1 # used for seasonal ar coefs
    min_sma=-0.2 # used for seasonal ar coefs
    
    
    def __init__(self, seed):
        # seed for reproducability
        np.random.seed(seed)
        
        # number of modes
        self.n_modes = self.mode_num_gen(1, mn=5)[0]
        
        # variances of each mode
        self.mode_vars = self.mode_vars_gen(self.n_modes, mn=0)
        
        # variance of noise mode
        self.noise_var_median_scale = self.noise_var_median_scale_gen(1, mn=0.5)[0]
        self.noise_var = self.noise_var_median_scale*np.median(self.mode_vars)
        
        # mode types
        self.mode_types = self.choose_mode_types(self.n_modes)
        
        # time series info
        self.time_series_info = self.collect_timeseries_info(self.mode_types)
        
        df = pd.DataFrame({
            'type':['noise']+list(self.mode_types),
            'variance':[self.noise_var]+list(self.mode_vars),
            'timeseries_info':[None]+list(self.time_series_info),
        })
        self.summary = df
        
    @classmethod
    def mode_num_gen(cls, n, mn=5): 
        ps = np.random.poisson(8, size=n)
        rn = sum(ps<mn)
        if rn>0:
            ps[ps<mn]=cls.mode_num_gen(rn, mn=mn)
        return ps
    
    @classmethod
    def mode_vars_gen(cls, n, mn=0): 
        ps = np.random.gamma(3, 5, n)
        rn = sum(ps<mn)
        if rn>0:
            ps[ps<mn]=cls.mode_vars_gen(rn, mn=mn)
        return ps
        
    @classmethod
    def noise_var_median_scale_gen(cls, n, mn=0.5): 
        k =1 # shape
        mean=1
        theta= mean/k #scale
        ps = np.random.gamma(k, scale=theta, size=n)
        rn = sum(ps<mn)
        if rn>0:
            ps[ps<mn]=cls.noise_var_median_scale_gen(rn, mn=mn)
        return ps
        
    @classmethod
    def period_gen(cls, n, mn=8): 
        ps = np.random.rayleigh(60, size=n)
        rn = sum(ps<mn)
        if rn>0:
            ps[ps<mn]=cls.period_gen(rn, mn=mn)
        return ps
    
    def choose_mode_types(self, n_modes):
        mode_types, probs = zip( *[(k,v) for k,v in self.mode_type_probabilities.items()] )
        chosen_modes = np.random.choice(mode_types, size=n_modes, replace=True, p=probs)
        return chosen_modes
    
    @classmethod
    def _linear_ar_params_gen(cls):
        return list([np.random.random()*(cls.max_ar-cls.min_ar)+cls.min_ar])
    
    @classmethod
    def _linear_ma_params_gen(cls):
        return list([np.random.random()*(cls.max_ma-cls.min_ma)+cls.min_ma])
    
    def _linear_mode_timeseries_gen(self):
        '''
        returns :
            Tuple of
            (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        '''
        is_seasonal = np.random.random() < self.p_L_seasonal
        
        ar_params_list = self._linear_ar_params_gen()
        ma_params_list = self._linear_ma_params_gen()
        p = int(self.period_gen(1)[0]) if is_seasonal else 1
        sar_params_list = self._linear_ar_params_gen() if is_seasonal else []
        sma_params_list = self._linear_ma_params_gen() if is_seasonal else []
        
        return (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        
    @classmethod 
    def _periodic_ar_params_gen(cls):
        return list([np.random.random()*(cls.max_sar-cls.min_sar)+cls.min_sar])
    
    @classmethod
    def _periodic_ma_params_gen(cls):
        return list([np.random.random()*(cls.max_sma-cls.min_sma)+cls.min_sma])
    
    def _periodic_mode_timeseries_gen(self):
        '''
        returns :
            Tuple of
            (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        '''
        is_seasonal = np.random.random() < self.p_P_seasonal
        
        ar_params_list = self._periodic_ar_params_gen()
        ma_params_list = self._periodic_ma_params_gen()
        p = int(self.period_gen(1)[0])
        sar_params_list = self._periodic_ar_params_gen() if is_seasonal else []
        sma_params_list = self._periodic_ma_params_gen() if is_seasonal else []
        
        return (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        
    @staticmethod
    def _nonlinear_ar_params_gen():
        return list([np.random.random()])
    
    @staticmethod
    def _nonlinear_ma_params_gen():
        return list([np.random.random()])
    
    def _nonlinear_mode_timeseries_gen(self):
        '''
        returns :
            Tuple of
            (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        '''
        is_seasonal = np.random.random() < self.p_NL_seasonal
        
        ar_params_list = self._nonlinear_ar_params_gen()
        ma_params_list = self._nonlinear_ma_params_gen()
        p = self.period_gen(1)[0] if is_seasonal else 1
        sar_params_list = self._nonlinear_ar_params_gen() if is_seasonal else []
        sma_params_list = self._nonlinear_ma_params_gen() if is_seasonal else []
        
        return (ar_params_list, ma_params_list, p, sar_params_list, sma_params_list)
        
    
    def _collect_single_timeseries_info(self, mode_type):
        if mode_type in ['linear_dense', 'linear_sparse']:
            return self._linear_mode_timeseries_gen()
        elif mode_type=='non_linear':
            return self._nonlinear_mode_timeseries_gen()
        else:
            return self._periodic_mode_timeseries_gen()
        
    def collect_timeseries_info(self, mode_types):
        ts_info = [self._collect_single_timeseries_info(mt) for mt in mode_types]
        return ts_info
        


if __name__=='__main__':    
    import matplotlib.pyplot as plt
    c = pseudoclim(None)
    print(c.summary)
    
    ##############################################
    # distribution of variance and number of modes
    ##############################################
    print('\n'*5)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    df = pd.DataFrame({'variance': pseudoclim.mode_vars_gen(10000)})
    df.plot(kind='kde', ax=ax1)
    ax1.set_xlabel('variance')
    ax1.set_xlim(0,50)
    print('var_distribution', ['{}:{:.2f}'.format(f.__name__, f(df.values)) for f in [np.min, np.mean, np.median, np.max]])

    df = pd.DataFrame({'y':pseudoclim.mode_num_gen(10000)})
    (df.groupby(by='y').y.count()/len(df)).plot(ax=ax2, label='number of modes', style='.')
    ax2.set_xlabel('number of modes')
    ax2.set_ylabel('Density')

    plt.tight_layout()
    plt.show()
    
    
    ##############################################
    # distribution of variance and number of modes
    ##############################################
    print('\n'*5)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))

    df = pd.DataFrame({'period':pseudoclim.period_gen(10000)})
    df.plot(kind='box', ax=ax1)
    ax1.set_xlabel('period')
    ax1.set_xlim([0, None])
    print('period_distribution', ['{}:{:.2f}'.format(f.__name__, f(df.values)) for f in [np.min, np.mean, np.median, np.max]])

    df = pd.DataFrame({'noise_median_scale':pseudoclim.noise_var_median_scale_gen(10000)})
    df.plot.kde(bw_method=0.03, ax=ax2)
    ax2.set_xlabel('noise_median_scale')
    ax2.set_xlim([0, 8])
    print('noise_median_scale', ['{}:{:.2f}'.format(f.__name__, f(df.values)) for f in [np.min, np.mean, np.median, np.max]])

    plt.tight_layout()
    plt.show()