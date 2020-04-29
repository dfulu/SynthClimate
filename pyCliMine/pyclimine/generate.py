###############
## TO DO
###############

# Tidy up some code in places to remove duplication as much as possible
# Implement save function

###############

import datetime
import cf_units
import re
import os

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from seaborn import heatmap

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pyshtools import SHGrid

from pyclimine.visualisation import animate_climate_fields
from pyclimine.utils import area_weighted_variance, flatten_except, check_make_dir

#######################################
## Kernel functions
#######################################

def calculate_central_angle(coords1, coords2):
    '''Calculate central angle distance between sets of coordinates
    Input and output in degrees.
    Args:
        coords1: (n x 2 array) n sets of lat and lon coords in degrees
        coords2: (n x 2 array) n sets of lat and lon coords in degrees
    returns:
        central angle between pairs of coords, coords1 and coords2, in degrees
        '''
    lats1_radians, lons1_radians = np.deg2rad(coords1[:,0]), np.deg2rad(coords1[:,1])
    lats2_radians, lons2_radians = np.deg2rad(coords2[:,0]), np.deg2rad(coords2[:,1])
    lats1_radians = lats1_radians[np.newaxis,:]
    lons1_radians = lons1_radians[np.newaxis,:]
    lats2_radians = lats2_radians[np.newaxis,:]
    lons2_radians = lons2_radians[np.newaxis,:]
    distance_radians = 2*np.arcsin(np.sqrt((np.sin((lats1_radians-lats2_radians.T)/2)**2)     \
                  + np.cos(lats1_radians)*np.cos(lats2_radians.T)*(np.sin((lons1_radians-lons2_radians.T)/2)**2)) - 1e-15 )
    return np.rad2deg(distance_radians)

def spherical_space_gauss_kernel_function(X1, X2, ell, sigma_f):
    '''Generates Gaussian kernel for points X1 X2 using distance metric of
    central angle.
    Args:
        X1: (n x 2 array) n sets of lat and lon coords in degrees
        X2: (n x 2 array) n sets of lat and lon coords in degrees
        ell: Characteristic width of Gaussian curve in degrees
        sigma_f: Amplitude
    returns:
        Gaussian kernel as array
        '''
    return sigma_f**2 * np.exp(-0.5*(calculate_central_angle(X1, X2)/ell)**2)

def phase_gauss_kernel_function(X1, X2, ell):
    '''Generates Gaussian kernel for points X1 X2 using euclidean distance metric.
    Args:
        X1: (array) euclidean coords
        X2: (array) euclidean coords
        ell: Characteristic width of Gaussian curve
    returns:
        Gaussian kernel as array
        '''
    X1, X2 = X1/(np.sqrt(2)*ell), X2/(np.sqrt(2)*ell)
    return np.exp((np.dot(X1,(2*X2.T))-np.sum(X1*X1,1)[:,None]) - np.sum(X2*X2,1)[None,:])

def cyclic_phase_gauss_kernel_function(X1, X2, ell):
    '''Generates periodic Gaussian kernel for points X1 X2.
    Args:
        X1: (array) euclidean coords
        X2: (array) euclidean coords
        ell: Characteristic width of Gaussian curve
    returns:
        Gaussian kernel as array
        '''
    return np.exp(-2/ell**2 * np.sin((X1-X2.T)*np.pi)**2)

#######################################
## Distribution draw functions
#######################################

def normal_dist_draw_func(N):
    """Generate N draws from a normal distribution
    Summary:
        Normal distribution:"""
    distribution_draw = np.random.randn(N)
    return distribution_draw

def _n_sparse_modes(n, mn=1): 
    """helper function for below
    Draw from poisson distribution with redraw clause"""
    ps = np.random.poisson(1.5, size=n)
    rn = sum(ps<mn)
    if rn>0:
        ps[ps<mn]=_n_sparse_modes(rn, mn=mn)
    return ps

def sparse_dist_draw_func(N):
    """Summary:Draw from possion distributition to select how many modes to mix.
    Then draw from normal for mixing strength:"""
    distribution_draw = np.random.randn(N)
    n_modes = _n_sparse_modes(1)[0]
    U_N = 0.8*N
    L_N=0 # don't select decomp modes below this index. Too sparse spatially
    selection = np.zeros(N)
    selection[np.random.randint(L_N, U_N, size=n_modes)]=1
    distribution_draw = distribution_draw*selection
    return distribution_draw

def parse_docstring_for_summary(fn):
    docstring = fn.__doc__
    try:
        summarystring = re.sub(' +', ' ',
           re.search(r'Summary:([^:]*)', docstring).group(1)
          ).replace('\n', '')
    except:
        summarystring = 'NO SUMMARY AVAILABLE'

    return fn.__name__ + ' : ' + summarystring

#######################################
## Time series draw functions
#######################################

def simulate_sarma(N, ar_params=[], ma_params=[], sar_params=[], sma_params=[], period=1, sigma=1.):
    """Generate a SARMA time series from given AR and MA parameters
    Args:
        N: (int) Length of time series to generate
        ar_params: (list) Autoregressive coefficients
        ma_params: (list) Moving average coefficients
        sar_params: (list) Seasonal autoregressive coefficients
        sma_params: (list) Seasonal moving average coefficients
        period: (int) Seasonal period in time steps
        sigma: (float) Uncertainty

    """
    params = ar_params+ma_params+sar_params+sma_params+[sigma]

    mod = SARIMAX([],
                  order=(len(ar_params), 0, len(ma_params)),
                  seasonal_order=(len(sar_params), 0, len(sma_params), period))

    return mod.simulate(params, N)

def simulate_n_arma(N, n, ar_params=[], ma_params=[], sigma=1.):
    """Generate a SARMA time series from given AR and MA parameters
    Args:
        n: (int) Number of time series to create
        N: (int) Length of time series to generate
        ar_params: (list) Autoregressive coefficients
        ma_params: (list) Moving average coefficients
        sar_params: (list) Seasonal autoregressive coefficients
        sma_params: (list) Seasonal moving average coefficients
        period: (int) Seasonal period in time steps
        sigma: (float) Uncertainty

    """
    ar_op = np.array([ar_params])[:,::-1]
    ma_op = np.array([np.hstack([1,np.array(ma_params)])[::-1]])
    ar_pad = len(ar_params)
    ma_pad = len(ma_params)
    pad = max(ar_pad, ma_pad)
    shocks = np.random.normal(0,sigma,(n,N+pad))
    data = np.zeros((n,N+pad))
    for i in range(pad,N+pad):
        p1 = (ar_op*data[:,i-ar_pad:i]).sum(axis=1)
        p2 =  (ma_op*shocks[:,i-ma_pad:i+1]).sum(axis=1)
        data[:,i]= p1+p2
    return data[:,pad:]

def sigmoid_function(x, a=1.):
    return(1.+np.exp(-x/a))**-1

## default time series functions

def default_timeseries_fn(n_T):
    """Default function for creating a time series of length n_T
    Summary:
        ARMA(2,1) model with ar coefs [0.8, 0.1], ma_coefs [0.338], sigma 1.:"""
    ts = simulate_sarma(n_T, ar_params=[0.8, 0.1], ma_params=[0.338])
    ts/=ts.var()
    return ts

def default_non_linear_timeseries_fn(n_T):
    """Default function for creating a time series of length n_T for
    non-linear modes.
    Summary:
        ARMA(2,1) model with ar coefs [0.8, 0.1], ma_coefs [0.338], sigma 0.3.
        Then run through a sigmoid function s(x).:"""
    ts = simulate_sarma(n_T, ar_params=[0.8, 0.1], ma_params=[0.338], sigma=0.3)
    ts = sigmoid_function(ts, a=1.)
    return ts

def default_non_linear_cyclic_timeseries_fn(n_T):
    """Default function for creating a time series of length n_T for
    non-linear modes.
    Summary:
        Phase increases smoothly from 0-1 with period 30 steps.
        Then repeats:"""
    P=30
    P = int(n_T/3) if n_T<P else P
    ts = (np.arange(n_T, dtype=float)/P)%1
    return ts

def default_noise_timeseries_fn(n_T):
    """Default function for creating a time series of length n_T for
    noise modes
    Summary:
        MA(3) model with ma coefs [0.3, -0.2, 0.1], sigma 1. Then normalised to unit
        variance.:"""
    ts = simulate_sarma(n_T, ar_params=[], ma_params=[0.3, -0.2, 0.1])
    ts/=ts.var()
    return ts

def default_traversing_mode_timeseries_fn(n_T):
    """Default function for creating a time series of length n_T for the
    phase of a traversing wave.
    Summary:
        Phase increases smoothly from 0-1 with period 30 steps.
        Then repeats:"""
    P=30
    P = int(n_T/3) if n_T<P else P
    ts = (np.arange(n_T, dtype=float)/P)%1
    return ts

def default_phase_to_lat_lon_path(phase_timeseries):
    """Default function for creating a lat-lon path from the
    phase series of a traveling wave
    Summary:
        lats offset is sinusoidal with amplitude 15deg and repeats
        once per phase.
        lons offset increases smoothly from 0-360 over one phase.:"""
    dlats = 15*np.sin(2*np.pi*phase_timeseries)
    dlons = 360*phase_timeseries
    return dlats, dlons

#######################################
## Data generating classes
#######################################


class modes_object(object):
    mode_type = 'none'
    
    def __init__(self,
                lats = np.arange(90,-91,-2.5),
                lons = np.arange(0,360,3.75),
                n_T = 100
    ):
        
        self.n_T = n_T
        self.lats = lats
        self.lons = lons
        self.coords_grid = np.stack(np.meshgrid(lats, lons)).T
        self.X_grid = self.coords_grid.reshape((np.product(self.coords_grid.shape[:-1]),2))

        self.N_grid = self.X_grid.shape[0]
        self.n_modes=0
        self.summary = pd.DataFrame(columns = ['mode_type', 'mode_variance',
                                               'spatial_kernel_info',
                                               'phase_kernel_info', 'kernel_draw_info',
                                               'timeseries_info', 'wave_path_info'])
        self.summary.index.name = 'mode_number'


    def plot_timeseries(self, sample=None, indexes=[]):
        if indexes == [] and sample is None:
            indexes = np.arange(self.n_modes)
        elif indexes == [] and sample is not None:
            indexes = np.sort(np.random.choice(
                np.arange(self.n_modes), size=sample,  replace=False))
        fig = plt.figure(figsize=(20,5*(len(indexes)//3+1)))
        for i, ind in enumerate(indexes):
            ax = plt.subplot(len(indexes)//3 + 1, 3, i+1)
            ax.set_title(self.mode_type + " time series {}".format(ind))
            plt.plot(np.arange(self.n_T), self.timeseries[ind])
        return fig

    def _update_summary(self, mode_indexes, spatial_kernel_ells=None,
                       kernel_sigmas=None, phase_kernel_ells=None,
                       kernel_draw_functions=None, timeseries_functions=None,
                       path_functions=None, mode_type=None,
                       variances=None):
        for i, ind in enumerate(mode_indexes):
            if ind not in self.summary.index:
                self.summary.loc[ind]=['---']*len(self.summary.columns)
            if kernel_draw_functions is not None:
                self.summary.loc[ind, 'kernel_draw_info'] = \
                    parse_docstring_for_summary(kernel_draw_functions[i])
            if timeseries_functions is not None:
                self.summary.loc[ind, 'timeseries_info'] = \
                    parse_docstring_for_summary(timeseries_functions[i])
            if (spatial_kernel_ells is not None) and (kernel_sigmas is not None):
                self.summary.loc[ind, 'spatial_kernel_info'] = \
                    'kernel_ells: {}\nkernel_sigmas: {}'.format(
                    spatial_kernel_ells, kernel_sigmas)
            if phase_kernel_ells is not None:
                self.summary.loc[ind, 'phase_kernel_info'] = \
                    'kernel_ells: {}'.format(phase_kernel_ells)
            if path_functions is not None:
                self.summary.loc[ind, 'wave_path_info'] = \
                    parse_docstring_for_summary(path_functions[i])
            if mode_type is not None:
                self.summary.loc[ind, 'mode_type'] = mode_type
            if timeseries_functions is not None:
                self.summary.loc[ind, 'mode_variance'] = variances[i]
    
    def _variance_modification(self, data, target_variance):
        var0 = area_weighted_variance(data, self.lats, time_ax = 0, lat_ax = 1)
        return data*(target_variance/var0)**0.5
    
    def project_to_xarray(self, flatten=False, t_ind_start=None, t_ind_stop=None):
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        time = np.arange(t_ind_start, t_ind_stop)
        
        if not flatten:
            data = self.project_modes(t_ind_start=t_ind_start, t_ind_stop=t_ind_stop)
            mode = np.arange(len(data))
            data = xr.DataArray(data, 
                                 coords=[mode, time, self.lats, self.lons], 
                                 dims=['mode', 'time', 'latitude', 'longitude'])
        else:
            data = self.flatten_modes(t_ind_start=t_ind_start, t_ind_stop=t_ind_stop)
            data = xr.DataArray(data, 
                                 coords=[time, self.lats, self.lons], 
                                 dims=['time', 'latitude', 'longitude'])
        return data
        

    def animate_data(self, indexes=[],
                     t_ind_start=0,
                     t_ind_stop=100,
                     flatten=True,
                     save_path='',
                     colorbar_clip_pct = 1,
                     share_colorbar = False):
        if flatten:
            data = self.flatten_modes(t_ind_start=t_ind_start,
                                      t_ind_stop=t_ind_stop)
            data = data.reshape((1,)+data.shape)
        else:
            data = self.project_modes(indexes=indexes,
                                      t_ind_start=t_ind_start,
                                      t_ind_stop=t_ind_stop)

        animate_climate_fields(data,
                               self.lats, self.lons,
                               suptitle='', fps=10,
                               save_path=save_path,
                               colorbar_clip_pct = colorbar_clip_pct,
                               share_colorbar = share_colorbar)


#######################################


class linear_modes_object(modes_object):
    mode_type = 'linear'
    
    @staticmethod
    def _kernel_functional(kernel_ells, kernel_sigmas):
        def kernel_function(X1, X2):
            return np.stack([spherical_space_gauss_kernel_function(X1, X2, 
                                ell=kernel_ells[i], sigma_f=kernel_sigmas[i])
                                for i  in range(len(kernel_ells))], axis=0).sum(axis=0)
        return kernel_function

    def add_linear_modes(self, n_modes, kernel_ells = [30], kernel_sigmas = [1],
                        data_distribution_draw_function = normal_dist_draw_func, 
                         err=1e-4):

        self.n_modes += n_modes

        self.data_distribution_draw_function = data_distribution_draw_function

        data_kernel_function = self._kernel_functional(kernel_ells, kernel_sigmas)

        # compute the kernel function. We can then chuck the data out
        self.K_grid = data_kernel_function(self.X_grid, self.X_grid) \
                        + err*np.eye(self.N_grid) 
        # above add jitter term to assure positive definite : Account for numerical error

        self.L_grid = np.linalg.cholesky(self.K_grid)

        modes = np.zeros((n_modes,) + self.coords_grid.shape[:2])

        for i in range(n_modes):
            modes[i] =  np.dot(self.L_grid, 
                data_distribution_draw_function(self.N_grid)) \
                .reshape(self.coords_grid.shape[:2])

        if hasattr(self, 'modes'):
            self.modes = np.concatenate([self.modes,modes], axis=0)
        else:
            self.modes = modes

        mode_indexes = np.arange(self.n_modes - n_modes, self.n_modes)
        self._update_summary(mode_indexes, spatial_kernel_ells=kernel_ells,
                       kernel_sigmas=kernel_sigmas,  mode_type=self.mode_type,
                       kernel_draw_functions=[data_distribution_draw_function]*n_modes)

    def resample_modes(self, mode_indexes=None):
        if mode_indexes is None:
            mode_indexes = np.arange(self.n_modes)
        for i in mode_indexes:
            self.modes[i] =  np.dot(self.L_grid, 
                        self.data_distribution_draw_function(self.N_grid))\
                        .reshape(self.coords_grid.shape[:2])

    def plot_modes(self):
        fig = plt.figure(figsize=(20,5*(self.n_modes//3+1)))
        for i in range(self.n_modes):
            ax = plt.subplot(self.n_modes//3 + 1, 3, i+1)
            ax.set_title(self.mode_type + " mode {}".format(i))
            f_grid = self.modes[i]
            heatmap(f_grid, square =True)
        return fig

    def implement_timeseries_functions(self, fns = [default_timeseries_fn], vrs=[1]):
        indexes = np.arange(self.n_modes)
        ts = np.zeros((self.n_modes, self.n_T))
        if len(fns)!=1 and len(fns)!=self.n_modes:
            raise ValueError('number of time series functions and n_modes do not match')
        elif len(fns)==1:
            fns = fns*self.n_modes
        if len(vrs)!=1 and len(vrs)!=self.n_modes:
            raise ValueError('number of variances and n_modes do not match')
        elif len(vrs)==1:
            vrs = vrs*self.n_modes
        for i in indexes:
            ts[i] = fns[i](self.n_T)
        self.variances = vrs
        self.timeseries = ts
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    def replace_timeseries(self, indexes=None, fns = [default_timeseries_fn], vrs=[1]):
        if indexes is None:
            indexes = np.arange(self.n_modes)
        if len(fns)!=1 and len(fns)!=len(indexes):
            raise ValueError('time series functions and indexes do not match')
        elif len(fns)==1:
            fns = fns*len(indexes)
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)
        for n,ind in enumerate(indexes):
            self.timeseries[ind] = fns[n](self.n_T)
            self.variances[ind]=vrs[n]
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    def project_modes(self, indexes=[], t_ind_start=None, t_ind_stop=None):
        '''Projects one or more modes against its time series.
        Args:
            indexes: list of int
        returns:
            array of individual mode projections with the time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        indexes = np.arange(self.n_modes) if indexes==[] else indexes
        data = np.zeros((len(indexes), t_ind_stop-t_ind_start,) + self.modes.shape[1:])
        for i, ind in enumerate(indexes):
            # projection without controlling variance
            X = (self.modes[ind][np.newaxis, :,:] *
                       self.timeseries[ind][t_ind_start:t_ind_stop, np.newaxis, np.newaxis])
            # control variance of mode
            data[i] = self._variance_modification(X, self.variances[ind])
        return data

    def flatten_modes(self, t_ind_start=None, t_ind_stop=None):
        '''Projects all the modes into a multivariate time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        data = np.zeros((t_ind_stop-t_ind_start,) + self.modes.shape[1:])
        for m,ts,v in zip(self.modes, self.timeseries, self.variances):
            X = m[np.newaxis, :,:] * ts[t_ind_start:t_ind_stop, np.newaxis, np.newaxis]
            data += self._variance_modification(X, v)
        return data
    
    def mode_snaps(self):
        """Method which returns a linear estimate/caricature of the modes.
        This will just be used for plotting comparisons between linear and 
        non-linear modes"""
        linear_modes = self.modes
        nfac = (linear_modes**2).sum(axis=(1,2))
        return linear_modes/(nfac[:, np.newaxis, np.newaxis]**0.5)

#######################################

class noise_modes_object(modes_object):
    mode_type = 'noise'
    
    @staticmethod
    def _kernel_functional(kernel_ells, kernel_sigmas):
        def kernel_function(X1, X2):
            return np.stack(
                [spherical_space_gauss_kernel_function(X1, X2, 
                                ell=kernel_ells[i], sigma_f=kernel_sigmas[i])
                                for i  in range(len(kernel_ells))],
                axis=0).sum(axis=0)
        return kernel_function

    def add_noise_modes(self, kernel_ells = [30, 15], kernel_sigmas = [1,1]):

        data_kernel_function = self._kernel_functional(kernel_ells, kernel_sigmas)

        self.K_grid = data_kernel_function(self.X_grid, self.X_grid) \
                        + 1e-4*np.eye(self.N_grid)

        self.L_grid = np.linalg.cholesky(self.K_grid)
        self.n_modes = self.L_grid.shape[0]
        self.modes = self.L_grid.reshape((self.n_modes,)+self.coords_grid.shape[:2])

        self._update_summary([0], spatial_kernel_ells=kernel_ells,
                       kernel_sigmas=kernel_sigmas,  mode_type=self.mode_type)

    def plot_modes(self, sample=3, indexes = []):
        if indexes == []:
            indexes = np.sort(np.random.choice(np.arange(self.n_modes), 
                                               size=sample, replace=False))
        fig = plt.figure(figsize=(20,5*(len(indexes)//3+1)))
        for pi, i in enumerate(indexes):
            ax = plt.subplot(len(indexes)//3 + 1, 3, pi+1)
            ax.set_title(self.mode_type+" mode {}".format(i))
            f_grid = self.modes[i]
            heatmap(f_grid, square =True)
        return fig

    def implement_timeseries_functions(self, fn = default_noise_timeseries_fn, vr=1):
        indexes = np.arange(self.n_modes)
        ts = np.zeros((self.n_modes, self.n_T))
        for i in indexes:
            ts[i] = fn(self.n_T)
        self.timeseries = ts
        self.variance = vr
        self._update_summary([0], timeseries_functions=[fn], variances=[vr])

    def replace_timeseries(self, fn= default_noise_timeseries_fn, vr=1):
        indexes = np.arange(self.n_modes)
        for i in indexes:
            self.timeseries[i] = fn(self.n_T)
            self.variance = vr
            self._update_summary([0], timeseries_functions=[fn], variances=[vr])

    def plot_timeseries(self, sample=3, indexes=[]):
        return super().plot_timeseries(sample=sample, indexes=indexes)
    
    def flatten_modes(self, t_ind_start=None, t_ind_stop=None):
        '''Projects all the noise modes into a multivariate time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        final_shape =  (t_ind_stop - t_ind_start,) + self.modes.shape[1:]
        data = np.matmul(self.timeseries[:,t_ind_start:t_ind_stop].T, 
                         flatten_except(self.modes, axis=(0,)))
        data = data.reshape(final_shape)
        data = self._variance_modification(data, self.variance)
        return data
    
    def project_modes(self, t_ind_start=None, t_ind_stop=None):
        '''See flatten_modes'''
        projection = self.flatten_modes(t_ind_start=t_ind_start, t_ind_stop=t_ind_stop)
        return projection.reshape((1,)+projection.shape)
    
    def mode_snaps(self):
        """Method which returns a linear estimate/caricature of the modes.
        This will just be used for plotting comparisons between linear and 
        non-linear modes"""
        linear_modes = self.modes.sum(axis=0, keepdims=True)
        nfac = (linear_modes**2).sum(axis=(1,2))
        return linear_modes/(nfac[:, np.newaxis, np.newaxis]**0.5)

#######################################

class non_linear_modes_object(modes_object):
    mode_type = 'non_linear'
    
    def __init__(self,
                lats = np.arange(90,-91,-2.5),
                lons = np.arange(0,360,3.75),
                phase_steps = 9,
                n_T = 100
    ):
        lats_phase = np.arange(90,-91,-7.5)
        lons_phase = np.arange(0,360,10)
        self.phase_steps = phase_steps
        super().__init__(lats = lats, lons = lons, n_T = n_T)

    @staticmethod
    def _kernel_functional(kernel_space_ells, kernel_space_sigmas, kernel_time_ells):

        def kernel_space_function(X1, X2):
            return np.stack([spherical_space_gauss_kernel_function(X1, X2,
                                ell=kernel_space_ells[i], sigma_f=kernel_space_sigmas[i])
                                for i  in range(len(kernel_space_ells))], axis=0).sum(axis=0)
        def kernel_time_function(T1, T2):
            return np.stack([phase_gauss_kernel_function(T1, T2, ell=kernel_time_ells[i])
                                for i  in range(len(kernel_time_ells))], axis=0).sum(axis=0)
        def kernel_function(X1, X2):
            return kernel_space_function(X1[:,:2], X2[:,:2])*kernel_time_function(X1[:,2:3], X2[:,2:3])

        return kernel_function

    @staticmethod
    def _interpolate_lat_lon(phase_field, lats_phase, lons_phase, lats, lons):

        if 360 not in lons_phase:
            lons_phase = np.hstack([lons_phase, 360])
            phase_field = np.concatenate([phase_field, phase_field[:,:,0:1]], axis=2)

        xr_phase_field = xr.DataArray(phase_field,
                                      [('phase', np.linspace(0,1,phase_field.shape[0])),
                                       ('lats', lats_phase),
                                       ('lons', lons_phase)])

        return xr_phase_field.interp(lats=lats, lons=lons).values

    def add_non_linear_modes(self, n_modes,
                        kernel_space_ells = [30],
                        kernel_sigmas = [1],
                        kernel_time_ells = [0.4],
                        data_distribution_draw_function = normal_dist_draw_func,
                        lats_phase = np.arange(90,-91,-7.5),
                        lons_phase = np.arange(0,360,10)):

        phase = np.linspace(0,1, self.phase_steps)

        self.lats_phase = lats_phase
        self.lons_phase = lons_phase

        self.n_modes += n_modes

        self.data_distribution_draw_function = normal_dist_draw_func

        data_kernel_function = self._kernel_functional(kernel_space_ells, 
                                                       kernel_sigmas, kernel_time_ells)

        coords_grid_phase = np.stack(np.meshgrid(lats_phase, lons_phase, phase)).T
        X_grid_phase = coords_grid_phase.reshape((np.product(coords_grid_phase.shape[:-1]), 3))
        N_grid_phase = X_grid_phase.shape[0]

        K_grid_phase = data_kernel_function(X_grid_phase, X_grid_phase) \
                        + 1e-4*np.eye(N_grid_phase)

        self.L_grid = np.linalg.cholesky(K_grid_phase)

        modes = np.zeros((n_modes, self.phase_steps)+self.coords_grid.shape[:2])

        for i in range(n_modes):
            distribution_draw = self.data_distribution_draw_function(N_grid_phase)
            f_grid_phase = np.dot(self.L_grid, distribution_draw).reshape(coords_grid_phase.shape[:3])
            filled_phase_field = self._interpolate_lat_lon(f_grid_phase, lats_phase, 
                                                           lons_phase, self.lats, self.lons)
            modes[i] = filled_phase_field

        if hasattr(self, 'modes'):
            self.modes = np.concatenate([self.modes,modes], axis=0)
        else:
            self.modes = modes

        indexes = np.arange(self.n_modes - n_modes, self.n_modes)
        self._update_summary(indexes, spatial_kernel_ells=kernel_space_ells,
                     kernel_sigmas=kernel_sigmas, phase_kernel_ells=kernel_time_ells,
                     kernel_draw_functions=[data_distribution_draw_function]*n_modes,
                     mode_type=self.mode_type)

    def resample_modes(self, mode_indexes=None):
        if mode_indexes is None:
            mode_indexes = np.arange(self.n_modes)

        for i in mode_indexes:
            distribution_draw = self.data_distribution_draw_function(self.L_grid.shape[0])
            f_grid_phase = np.dot(self.L_grid, distribution_draw).reshape((self.phase_steps, self.lats_phase.shape[0], self.lons_phase.shape[0]))
            filled_phase_field = self._interpolate_lat_lon(f_grid_phase, self.lats_phase, self.lons_phase, self.lats, self.lons)
            self.modes[i] = filled_phase_field
            return

    def plot_mode(self, n):
        fig = plt.figure(figsize=(7, 5*self.phase_steps))
        for p in range(self.phase_steps):
                ax = plt.subplot(self.phase_steps, 1, p+1)
                ax.set_title("mode {} : phase number {}".format(n, p))
                f_grid = self.modes[n][p]
                heatmap(f_grid, square =True)
        plt.tight_layout()
        return fig

    def plot_modes(self):
        fig = plt.figure(figsize=(7*self.n_modes, 5*self.phase_steps))
        for i in range(self.n_modes):
            for p in range(self.phase_steps):
                ax = plt.subplot(self.phase_steps, self.n_modes,  self.n_modes*p + i + 1)
                ax.set_title(self.mode_type+" mode {} : phase number {}".format(i, p))
                f_grid = self.modes[i][p]
                heatmap(f_grid, square =True)
        plt.tight_layout()
        return fig

    def implement_timeseries_functions(self, fns = [default_non_linear_timeseries_fn], vrs=[1]):
        indexes = np.arange(self.n_modes)
        ts = np.zeros((self.n_modes, self.n_T))
        if len(fns)!=1 and len(fns)!=self.n_modes:
            raise ValueError('time series functions and n_modes do not match')
        elif len(fns)==1:
            fns = fns*self.n_modes
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)
        for i in indexes:
            ts[i] = fns[i](self.n_T)
        self.variances=vrs
        self.timeseries = ts
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    def replace_timeseries(self, indexes=None, fns = [default_non_linear_timeseries_fn], vrs=[1]):
        if indexes is None:
            indexes = np.arange(self.n_modes)
        if len(fns)!=1 and len(fns)!=len(indexes):
            raise ValueError('time series functions and indexes do not match')
        elif len(fns)==1:
            fns = fns*len(indexes)
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)
        for n,ind in enumerate(indexes):
            self.timeseries[ind] = fns[n](self.n_T)
            self.variances[ind]=vrs[n]
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    @staticmethod
    def _project_mode(timeseries, mode):
        xr_phase_field = xr.DataArray(mode,
                                      [('phase', np.linspace(0,1,mode.shape[0])),
                                       ('coord1', np.arange(mode.shape[1])),
                                       ('coord2', np.arange(mode.shape[2]))])
        return xr_phase_field.interp(phase=timeseries, method='cubic').values

    def project_modes(self, indexes=[], t_ind_start=None, t_ind_stop=None):
        '''Projects one or more modes against its time series.
        Args:
            indexes: list of int
        returns:
            array of individual mode projections with the time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        indexes = np.arange(self.n_modes) if indexes==[] else indexes
        data = np.zeros((len(indexes), t_ind_stop-t_ind_start) + self.modes.shape[-2:])
        for i, ind in enumerate(indexes):
            X = self._project_mode(self.timeseries[ind][t_ind_start:t_ind_stop], self.modes[ind])
            data[i] = self._variance_modification(X, self.variances[ind])
        return data
    
    def flatten_modes(self, t_ind_start=None, t_ind_stop=None):
        '''Projects all the modes into a multivariate time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        data = np.zeros((t_ind_stop-t_ind_start,) + self.modes.shape[-2:])
        for m,ts,v in zip(self.modes, self.timeseries, self.variances):
            X = self._project_mode(ts[t_ind_start:t_ind_stop], m)
            data += self._variance_modification(X, v)
        return data
    
    def animate_non_linear_modes(self, indexes=[], save_path='', figsize=None, 
                                 frames=None, colorbar_clip_pct = 1):
        indexes = np.arange(self.n_modes) if indexes == [] else indexes
        subtitles = [self.mode_type+" mode {}".format(i) 
                     for i in range(self.n_modes)]
        if frames is None: frames = self.phase_steps
        t = np.linspace(0,1,frames, endpoint=self.mode_type=='non_linear')
        data = np.zeros((len(indexes), frames) + self.modes.shape[-2:])
        for i, ind in enumerate(indexes):
            X = self._project_mode(t, self.modes[ind])
            data[i] = self._variance_modification(X, self.variances[ind])
        animate_climate_fields(data,
                               self.lats, self.lons,
                               suptitle='', fps=frames/10,
                               save_path=save_path,
                               subtitles=subtitles,
                               figsize=figsize,
                               colorbar_clip_pct = colorbar_clip_pct,
                               share_colorbar = False)
    
    def mode_snaps(self):
        """Method which returns a linear estimate/caricature of the modes.
        This will just be used for plotting comparisons between linear and 
        non-linear modes"""
        n = self.modes.shape[1]//2
        linear_modes = self.modes[:,-n:].mean(axis=1) - self.modes[:,:n].mean(axis=1)
        nfac = (linear_modes**2).sum(axis=(1,2))
        return linear_modes/(nfac[:, np.newaxis, np.newaxis]**0.5)

#######################################
    
class non_linear_cyclic_modes_object(non_linear_modes_object):
    mode_type = 'non_linear_cyclic'

    @staticmethod
    def _kernel_functional(kernel_space_ells, kernel_space_sigmas, kernel_time_ells):

        def kernel_space_function(X1, X2):
            return np.stack([spherical_space_gauss_kernel_function(X1, X2,
                                ell=kernel_space_ells[i], sigma_f=kernel_space_sigmas[i])
                                for i  in range(len(kernel_space_ells))], axis=0).sum(axis=0)
        def kernel_time_function(T1, T2):
            return np.stack([cyclic_phase_gauss_kernel_function(T1, T2, ell=kernel_time_ells[i])
                                for i  in range(len(kernel_time_ells))], axis=0).sum(axis=0)
        def kernel_function(X1, X2):
            return kernel_space_function(X1[:,:2], X2[:,:2])*kernel_time_function(X1[:,2:3], X2[:,2:3])

        return kernel_function

    def add_non_linear_modes(self, n_modes,
                        kernel_space_ells = [30],
                        kernel_sigmas = [1],
                        kernel_time_ells = [1],
                        data_distribution_draw_function = normal_dist_draw_func,
                        lats_phase = np.arange(90,-91,-7.5),
                        lons_phase = np.arange(0,360,10)):

        phase = np.linspace(0,1, self.phase_steps, endpoint=False)

        self.lats_phase = lats_phase
        self.lons_phase = lons_phase

        self.n_modes += n_modes

        self.data_distribution_draw_function = normal_dist_draw_func

        data_kernel_function = self._kernel_functional(kernel_space_ells, 
                                                       kernel_sigmas, kernel_time_ells)

        coords_grid_phase = np.stack(np.meshgrid(lats_phase, lons_phase, phase)).T
        X_grid_phase = coords_grid_phase.reshape((np.product(coords_grid_phase.shape[:-1]), 3))
        N_grid_phase = X_grid_phase.shape[0]

        K_grid_phase = data_kernel_function(X_grid_phase, X_grid_phase) \
                        + 1e-4*np.eye(N_grid_phase)

        self.L_grid = np.linalg.cholesky(K_grid_phase)

        modes = np.zeros((n_modes, self.phase_steps)+self.coords_grid.shape[:2])

        for i in range(n_modes):
            distribution_draw = self.data_distribution_draw_function(N_grid_phase)
            f_grid_phase = np.dot(self.L_grid, distribution_draw).reshape(coords_grid_phase.shape[:3])
            filled_phase_field = self._interpolate_lat_lon(f_grid_phase, lats_phase, 
                                                           lons_phase, self.lats, self.lons)
            modes[i] = filled_phase_field

        if hasattr(self, 'modes'):
            self.modes = np.concatenate([self.modes,modes], axis=0)
        else:
            self.modes = modes

        indexes = np.arange(self.n_modes - n_modes, self.n_modes)
        self._update_summary(indexes, spatial_kernel_ells=kernel_space_ells,
                     kernel_sigmas=kernel_sigmas, phase_kernel_ells=kernel_time_ells,
                     kernel_draw_functions=[data_distribution_draw_function]*n_modes,
                     mode_type=self.mode_type)

    def resample_modes(self, mode_indexes=None):
        if mode_indexes is None:
            mode_indexes = np.arange(self.n_modes)

        for i in mode_indexes:
            distribution_draw = self.data_distribution_draw_function(self.L_grid.shape[0])
            f_grid_phase = np.dot(self.L_grid, distribution_draw).reshape((self.phase_steps, self.lats_phase.shape[0], self.lons_phase.shape[0]))
            filled_phase_field = self._interpolate_lat_lon(f_grid_phase, self.lats_phase, self.lons_phase, self.lats, self.lons)
            self.modes[i] = filled_phase_field
            return

    def implement_timeseries_functions(self, fns = [default_non_linear_cyclic_timeseries_fn], vrs=[1]):
        indexes = np.arange(self.n_modes)
        ts = np.zeros((self.n_modes, self.n_T))
        if len(fns)!=1 and len(fns)!=self.n_modes:
            raise ValueError('time series functions and n_modes do not match')
        elif len(fns)==1:
            fns = fns*self.n_modes
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)
        for i in indexes:
            ts[i] = fns[i](self.n_T)
        self.variances=vrs
        self.timeseries = ts
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    def replace_timeseries(self, indexes=None, fns = [default_non_linear_cyclic_timeseries_fn], vrs=[1]):
        if indexes is None:
            indexes = np.arange(self.n_modes)
        if len(fns)!=1 and len(fns)!=len(indexes):
            raise ValueError('time series functions and indexes do not match')
        elif len(fns)==1:
            fns = fns*len(indexes)
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)
        for n,ind in enumerate(indexes):
            self.timeseries[ind] = fns[n](self.n_T)
            self.variances[ind]=vrs[n]
        self._update_summary(indexes, timeseries_functions=fns, variances=vrs)

    @staticmethod
    def _project_mode(timeseries, mode):
        looped_mode = np.concatenate([mode[-1:], mode, mode[:2]])
        phase_step = 1./mode.shape[0]
        xr_phase_field = xr.DataArray(looped_mode,
                                      [('phase', np.arange(0-phase_step, 1+phase_step, phase_step)),
                                       ('coord1', np.arange(mode.shape[1])),
                                       ('coord2', np.arange(mode.shape[2]))])
        return xr_phase_field.interp(phase=timeseries, method='cubic').values
    
#######################################

class moving_wave_mode_object(modes_object):
    mode_type = 'moving_wave'
    
    @staticmethod
    def _kernel_functional(kernel_ells, kernel_sigmas):
        def kernel_function(X1, X2):
            return np.stack([spherical_space_gauss_kernel_function(X1, X2, 
                                ell=kernel_ells[i], sigma_f=kernel_sigmas[i])
                                for i  in range(len(kernel_ells))], axis=0).sum(axis=0)
        return kernel_function

    def add_traversing_modes(self, n_modes, kernel_ells = [30], kernel_sigmas = [1],
                        data_distribution_draw_function = sparse_dist_draw_func,
                        n_sample = (60,120)):

        self.n_modes += n_modes

        self.regulated_lats = np.linspace(90,-90, n_sample[0], endpoint=False)
        self.regulated_lons = np.linspace(0, 360, n_sample[1], endpoint=False)
        self.regulated_coords_grid = np.stack(np.meshgrid(self.regulated_lats, self.regulated_lons)).T
        self.regulated_X_grid = self.regulated_coords_grid.reshape((np.product(self.regulated_coords_grid.shape[:-1]),2))

        self.regulated_N_grid = self.regulated_X_grid.shape[0]

        self.data_distribution_draw_function = data_distribution_draw_function

        data_kernel_function = self._kernel_functional(kernel_ells, kernel_sigmas)

        # compute the kernel function. We can then chuck the data out
        self.regulated_K_grid = data_kernel_function(self.regulated_X_grid, self.regulated_X_grid) \
                        + 1e-4*np.eye(self.regulated_N_grid) # add jitter term to assure positive definite : Account for numerical error

        self.regulated_L_grid = np.linalg.cholesky(self.regulated_K_grid)

        modes = np.zeros((n_modes,) + self.coords_grid.shape[:2])
        regulated_modes = np.zeros((n_modes,) + self.regulated_coords_grid.shape[:2])
        spherical_harmonics = np.zeros(n_modes, dtype=object)

        for i in range(n_modes):
            # create modes with regular grid so that it can be expressed using spherical harmoics package
            regulated_modes[i] =  np.dot(self.regulated_L_grid, data_distribution_draw_function(self.regulated_N_grid))\
                                    .reshape(self.regulated_coords_grid.shape[:2])
            # decompose into spherical harmonics
            spherical_harmonics[i] = SHGrid.from_array(regulated_modes[i]).expand()
            # project from spherical harmonics onto desired grid
            modes[i] = spherical_harmonics[i].expand(lat=self.X_grid[:,0], lon=self.X_grid[:,1])\
                                             .reshape(self.coords_grid.shape[:2])

        if hasattr(self, 'modes'):
            self.modes = np.concatenate([self.modes,modes], axis=0)
            self.regulated_modes = np.concatenate([self.regulated_modes,regulated_modes], axis=0)
            self.spherical_harmonics = np.concatenate([self.spherical_harmonics,spherical_harmonics], axis=0)

        else:
            self.modes = modes
            self.regulated_modes = regulated_modes
            self.spherical_harmonics = spherical_harmonics

        indexes = np.arange(self.n_modes-n_modes, self.n_modes)
        self._update_summary(indexes, spatial_kernel_ells=kernel_ells, mode_type=self.mode_type,
                     kernel_sigmas=kernel_sigmas, kernel_draw_functions=[data_distribution_draw_function]*n_modes,
                     )


    def resample_modes(self, mode_indexes=None):
        if mode_indexes is None:
            mode_indexes = np.arange(self.n_modes)
        for i in mode_indexes:
            # create modes with regular grid so that it can be expressed using spherical harmoics package
            self.regulated_modes[i] =  np.dot(self.regulated_L_grid, self.data_distribution_draw_function(self.regulated_N_grid))\
                                    .reshape(self.regulated_coords_grid.shape[:2])
            # decompose into spherical harmonics
            self.spherical_harmonics[i] = SHGrid.from_array(self.regulated_modes[i]).expand()
            # project from spherical harmonics onto desired grid
            self.modes[i] = self.spherical_harmonics[i].expand(lat=self.X_grid[:,0], lon=self.X_grid[:,1])\
                                             .reshape(self.coords_grid.shape[:2])

    def plot_modes(self):
        fig = plt.figure(figsize=(20,5*(self.n_modes//3+1)))
        for i in range(self.n_modes):
            ax = plt.subplot(self.n_modes//3 + 1, 3, i+1)
            ax.set_title(self.mode_type+" mode {}".format(i))
            f_grid = self.modes[i]
            heatmap(f_grid, square =True)
        return fig

    def implement_timeseries_functions(self,
                                       fns = [default_traversing_mode_timeseries_fn],
                                       phase_to_lat_lon_fns = [default_phase_to_lat_lon_path],
                                       vrs=[1]):
        indexes = np.arange(self.n_modes)
        ts = np.zeros((self.n_modes, self.n_T))
        dlats = np.zeros((self.n_modes, self.n_T))
        dlons = np.zeros((self.n_modes, self.n_T))

        if len(fns)!=1 and len(fns)!=self.n_modes:
            raise ValueError('time series functions and n_modes do not match')
        elif len(fns)==1:
            fns = fns*self.n_modes

        if len(phase_to_lat_lon_fns)!=1 and len(phase_to_lat_lon_fns)!=self.n_modes:
            raise ValueError('path functions and n_modes do not match')
        elif len(phase_to_lat_lon_fns)==1:
            phase_to_lat_lon_fns = phase_to_lat_lon_fns*self.n_modes
            
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)

        for i in indexes:
            ts[i] = fns[i](self.n_T)
            dlats[i], dlons[i] = phase_to_lat_lon_fns[i](ts[i])

        self.timeseries = ts
        self.variances=vrs
        self.dlats = dlats
        self.dlons = dlons

        self._update_summary(indexes, timeseries_functions=fns,
                             path_functions=phase_to_lat_lon_fns,
                             variances=vrs)

    def replace_timeseries(self, indexes=None,
                           fns = [default_traversing_mode_timeseries_fn],
                           phase_to_lat_lon_fns = [default_phase_to_lat_lon_path]):
        indexes = np.arange(self.n_modes) if indexes is None else indexes
        if len(fns)!=1 and len(fns)!=len(indexes):
            raise ValueError('time series functions and indexes do not match')
        if len(fns)==1:
            fns = fns*len(indexes)

        if len(phase_to_lat_lon_fns)!=1 and len(phase_to_lat_lon_fns)!=len(indexes):
            raise ValueError('path functions and n_modes do not match')
        if len(phase_to_lat_lon_fns)==1:
            phase_to_lat_lon_fns = phase_to_lat_lon_fns*len(indexes)
            
        if len(vrs)!=1 and len(vrs)!=len(indexes):
            raise ValueError('number of variances and indexes do not match')
        elif len(vrs)==1:
            vrs = vrs*len(indexes)

        for n,ind in enumerate(indexes):
            self.timeseries[ind] = fns[n](self.n_T)
            self.dlats[ind], self.dlons[ind] = phase_to_lat_lon_fns[n](self.timeseries[ind])
            self.variances[ind]=vrs[n]

        self._update_summary(indexes, timeseries_functions=fns,
                             path_functions=phase_to_lat_lon_fns,
                             variances=vrs)

    def plot_path(self, sample=None, indexes=[]):
        if indexes == [] and sample is None:
            indexes = np.arange(self.n_modes)
        elif indexes == [] and sample is not None:
            indexes = np.sort(np.random.choice(np.arange(self.n_modes), size=sample,  replace=False))
        fig = plt.figure(figsize=(20,5*(len(indexes)//3+1)))
        for i, ind in enumerate(indexes):
            ax = plt.subplot(len(indexes)//3 + 1, 3, i+1)
            ax.set_title(self.mode_type+" - path of mode {}".format(ind))
            plt.scatter(self.dlons[ind], self.dlats[ind])
            plt.xlabel('longitude offset')
            plt.ylabel('latitude offset')
        return fig

    def _project_mode(self,spherical_harmonic, dlats, dlons):
        data = np.zeros((len(dlats),) + self.coords_grid.shape[:2])
        for i, (dlat,dlon) in enumerate(zip(dlats, dlons)):
            data[i] = spherical_harmonic.rotate(0, -dlat, -dlon, degrees=True, convention='x')\
                                        .expand(lat=self.X_grid[:,0], lon=self.X_grid[:,1])\
                                        .reshape(self.coords_grid.shape[:2])
        return data

    def project_modes(self, indexes=[], t_ind_start=None, t_ind_stop=None):
        '''Projects one or more modes against its time series.
        Args:
            indexes: list of int
        returns:
            array of individual mode projections with the time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        indexes = np.arange(self.n_modes) if indexes==[] else indexes
        data = np.zeros((len(indexes), t_ind_stop-t_ind_start,) + self.coords_grid.shape[:2])
        for i, ind in enumerate(indexes):
            X = self._project_mode(self.spherical_harmonics[ind],
                                         self.dlats[ind][t_ind_start:t_ind_stop],
                                         self.dlons[ind][t_ind_start:t_ind_stop])
            data[i] = self._variance_modification(X, self.variances[ind])
        return data

    def flatten_modes(self, t_ind_start=None, t_ind_stop=None):
        '''Projects all the modes into a multivariate time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        data = np.zeros((t_ind_stop-t_ind_start,) + self.coords_grid.shape[:2])
        for i in range(self.n_modes):
            X = self._project_mode(self.spherical_harmonics[i],
                                       self.dlats[i][t_ind_start:t_ind_stop],
                                       self.dlons[i][t_ind_start:t_ind_stop])
            data += self._variance_modification(X, self.variances[i])
        return data
    
    def mode_snaps(self):
        """Method which returns a linear estimate/caricature of the modes.
        This will just be used for plotting comparisons between linear and 
        non-linear modes"""
        linear_modes = self.modes
        nfac = (linear_modes**2).sum(axis=(1,2))
        return linear_modes/(nfac[:, np.newaxis, np.newaxis]**0.5)

#######################################

class mixed_modes_master(object):
    def __init__(self,
                lats = np.arange(90,-91,-2.5),
                lons = np.arange(0,360,3.75),
                n_T = 100
    ):
        self.text_note=''
        self.n_T = n_T
        self.lats = lats
        self.lons = lons
        self.coords_grid = np.stack(np.meshgrid(lats, lons)).T
        self.X_grid = self.coords_grid.reshape((np.product(self.coords_grid.shape[:-1]),2))
        self.N_grid = self.X_grid.shape[0]

    def initiate_linear_modes(self):
        self.linear_modes = linear_modes_object(
                        n_T = self.n_T,
                        lats = self.lats,
                        lons = self.lons)

    def initiate_non_linear_modes(self, phase_steps=9):
        self.non_linear_modes = non_linear_modes_object(
                                                        n_T = self.n_T,
                                                        lats=self.lats,
                                                        lons=self.lons,
                                                        phase_steps = phase_steps)
        
    def initiate_non_linear_cyclic_modes(self, phase_steps=9):
        self.non_linear_cyclic_modes = non_linear_cyclic_modes_object(
                                                        n_T = self.n_T,
                                                        lats=self.lats,
                                                        lons=self.lons,
                                                        phase_steps = phase_steps)

    def initiate_noise_modes(self):
        self.noise_modes = noise_modes_object(n_T = self.n_T, lats=self.lats, lons=self.lons)

    def initiate_wave_modes(self):
        self.wave_modes = moving_wave_mode_object(n_T = self.n_T, lats = self.lats, lons = self.lons)

    def plot_modes(self):
        self._apply_over_modes_objects(lambda ob: ob.plot_modes())

    def plot_timeseries(self):
        self._apply_over_modes_objects(lambda ob: ob.plot_timeseries())

    def _apply_over_modes_objects(self, fn, _return=True, apply_over_noise=True):
        output = []
        if hasattr(self, 'noise_modes')&apply_over_noise:
            output.append(fn(self.noise_modes))
        if hasattr(self, 'linear_modes'):
            output.append(fn(self.linear_modes))
        if hasattr(self, 'non_linear_modes'):
            output.append(fn(self.non_linear_modes))
        if hasattr(self, 'non_linear_cyclic_modes'):
            output.append(fn(self.non_linear_cyclic_modes))
        if hasattr(self, 'wave_modes'):
            output.append(fn(self.wave_modes))
        if _return: return output
        
    def flatten_modes(self, t_ind_start=None, t_ind_stop=None):
        '''Projects all the modes into a multivariate time series'''
        t_ind_start = 0 if t_ind_start is None else t_ind_start
        t_ind_stop = self.n_T if t_ind_stop is None else t_ind_stop
        data = np.zeros((t_ind_stop - t_ind_start, 
                         self.lats.shape[0], 
                         self.lons.shape[0]))

        def data_adder(modes_ob):
            nonlocal data
            data += modes_ob.flatten_modes(t_ind_start=t_ind_start, 
                                           t_ind_stop=t_ind_stop)
            
        self._apply_over_modes_objects(data_adder, _return=False)
        return data
    
    def project_modes(self, t_ind_start=None, t_ind_stop=None):
        project = lambda ob: ob.project_modes(t_ind_start=t_ind_start, 
                                             t_ind_stop=t_ind_stop)
        projections = self._apply_over_modes_objects(project, _return=True)
        projections = np.concatenate(projections, axis=0)
        return projections
    
    def project_to_xarray(self, **kwargs):
        return modes_object.project_to_xarray(self, **kwargs)
    
    def animate_data(self,
                     t_ind_start=0,
                     t_ind_stop=100,
                     flatten=True,
                     save_path='',
                     figsize=None,
                     colorbar_clip_pct = 1, 
                     share_colorbar=False):
        if flatten:
            data = self.flatten_modes(t_ind_start=t_ind_start,
                                      t_ind_stop=t_ind_stop)
            data = data.reshape((1,)+data.shape)
        else:
            data = self.project_modes(t_ind_start=t_ind_start,
                                      t_ind_stop=t_ind_stop)

        animate_climate_fields(data,
                               self.lats, self.lons,
                               suptitle='', fps=10,
                               save_path=save_path, 
                               figsize=figsize,
                               colorbar_clip_pct = colorbar_clip_pct,
                               share_colorbar=share_colorbar)
        
    def mode_snaps(self):
        retrieve_snaps = lambda ob: ob.mode_snaps()
        snaps = self._apply_over_modes_objects(retrieve_snaps, _return=True)
        snaps = np.concatenate(snaps, axis=0)
        return snaps

    @property
    def n_modes(self):
        return sum(self._apply_over_modes_objects(lambda ob: ob.n_modes, 
                                                  apply_over_noise=False))

    @property
    def summary(self):
        df = pd.concat(self._apply_over_modes_objects(lambda ob: ob.summary))
        df = df.reset_index()
        df.index.name = 'mixed_mode_number'
        return df
    
    def dump_plots(self, directory='', non_lin_gif=False):
        """Method to create and save all plots to showcase the generated
        data"""
        
        def save_plt_if(fig, filename):
            if directory=='':
                return
            else:
                check_make_dir(directory)
                fname = os.path.join(directory, filename+'.png')
                fig.savefig(fname)
                            
        if hasattr(self, 'noise_modes'):
            indexes = np.sort(
                        np.random.choice(
                            np.arange(self.noise_modes.n_modes), 
                            size=3, replace=False)
            )
            
            fig=self.noise_modes.plot_modes(indexes=indexes)
            save_plt_if(fig, 'noise_modes')
                            
            fig=self.noise_modes.plot_timeseries(indexes=indexes)
            save_plt_if(fig, 'noise_timeseries')
                            
        if hasattr(self, 'linear_modes'):
            fig=self.linear_modes.plot_modes()
            save_plt_if(fig, 'linear_modes')
                            
            fig=self.linear_modes.plot_timeseries()
            save_plt_if(fig, 'linear_timeseries')
                            
        if hasattr(self, 'non_linear_modes'):
            fig=self.non_linear_modes.plot_modes()
            save_plt_if(fig, 'non_linear_modes')
                            
            fig=self.non_linear_modes.plot_timeseries()
            save_plt_if(fig, 'non_linear_timeseries')
            
            if non_lin_gif:
                gname=os.path.join(directory, 'non_linear_modes')
                self.non_linear_modes\
                .animate_non_linear_modes(save_path=gname)

        if hasattr(self, 'non_linear_cyclic_modes'):
            fig=self.non_linear_cyclic_modes.plot_modes()
            save_plt_if(fig, 'non_linear_cyclic_modes')
                            
            fig=self.non_linear_cyclic_modes.plot_timeseries()
            save_plt_if(fig, 'non_linear_cyclic_timeseries')
            
            if non_lin_gif:
                gname=os.path.join(directory, 'non_linear_cyclic_modes')
                self.non_linear_cyclic_modes\
                .animate_non_linear_modes(save_path=gname)
            
        if hasattr(self, 'wave_modes'):
            fig=self.wave_modes.plot_modes()
            save_plt_if(fig, 'moving_wave_modes')
                            
            fig=self.wave_modes.plot_timeseries()
            save_plt_if(fig, 'moving_wave_timeseries')
                            
            fig=self.wave_modes.plot_path()
            save_plt_if(fig, 'moving_wave_paths')


#######################################

if __name__=='__main__':
    make_linear = False
    make_noise = False
    make_non_linear = False
    make_non_linear_cyclic = False
    make_wave = True
    make_mixed = False

    if make_linear:
        # create linear modes object and add two modes
        m_linear = linear_modes_object()
        m_linear.add_linear_modes(2)

        # plot the modes
        fig1 = m_linear.plot_modes()
        plt.show()

        # resample the modes and plot again
        m_linear.resample_modes()
        fig2 = m_linear.plot_modes()
        plt.show()

        # create a timeseries for the latent variables of the modes and plot
        m_linear.implement_timeseries_functions()
        fig3 = m_linear.plot_timeseries()
        plt.show()

        # resample the timeseries and plot
        m_linear.replace_timeseries(fns = [default_timeseries_fn], indexes=[1])
        fig4 = m_linear.plot_timeseries()
        plt.show()

        print(m_linear.summary)

    if make_noise:
        # create linear modes object and add two modes
        m_noise = noise_modes_object()
        m_noise.add_noise_modes()

        # plot the modes
        fig1 = m_noise.plot_modes(sample=2) 
        plt.show()

        # create a timeseries for the latent variables of the modes and plot
        m_noise.implement_timeseries_functions()
        fig3 = m_noise.plot_timeseries(indexes = [0,1,2])
        plt.show()

        # resample the timeseries and plot
        m_noise.replace_timeseries(fn = default_noise_timeseries_fn)
        fig4 = m_noise.plot_timeseries(indexes = [0,1,2])
        plt.show()

        print(m_noise.summary)

    if make_non_linear:
        # create linear modes object and add two modes
        m_non_linear = non_linear_modes_object()
        m_non_linear.add_non_linear_modes(2)

        # plot the modes
        fig1 = m_non_linear.plot_modes()
        plt.show()

        # resample the modes and plot again
        m_non_linear.resample_modes([1])
        fig2 = m_non_linear.plot_modes()
        plt.show()

        # create a timeseries for the latent variables of the modes and plot
        m_non_linear.implement_timeseries_functions()
        fig3 = m_non_linear.plot_timeseries()
        plt.show()

        # resample the timeseries and plot
        m_non_linear.replace_timeseries(fns = [default_non_linear_timeseries_fn])
        fig4 = m_non_linear.plot_timeseries()
        plt.show()

        print(m_non_linear.summary)
        
    if make_non_linear_cyclic:
        # create linear modes object and add two modes
        m_non_linear_cyclic = non_linear_cyclic_modes_object()
        m_non_linear_cyclic.add_non_linear_modes(2)

        # plot the modes
        fig1 = m_non_linear_cyclic.plot_modes()
        plt.show()

        # resample the modes and plot again
        m_non_linear_cyclic.resample_modes([1])
        fig2 = m_non_linear_cyclic.plot_modes()
        plt.show()

        # create a timeseries for the latent variables of the modes and plot
        m_non_linear_cyclic.implement_timeseries_functions()
        fig3 = m_non_linear_cyclic.plot_timeseries()
        plt.show()

        # resample the timeseries and plot
        m_non_linear_cyclic.replace_timeseries(fns = [default_non_linear_timeseries_fn])
        fig4 = m_non_linear_cyclic.plot_timeseries()
        plt.show()

        print(m_non_linear_cyclic.summary)


    if make_wave:
        # create linear modes object and add two modes
        m_wave = moving_wave_mode_object()
        m_wave.add_traversing_modes(1)

        # plot the modes
        fig1 = m_wave.plot_modes()
        plt.show()

        # resample the modes and plot again
        m_wave.resample_modes()
        fig2 = m_wave.plot_modes()
        plt.show()

        # create a timeseries for the latent variables of the modes and plot
        m_wave.implement_timeseries_functions()
        fig3 = m_wave.plot_timeseries()
        plt.show()

        print(m_wave.summary)

    if make_mixed:
        m_mixed = mixed_modes_master(
                lats = np.arange(90,-91,-2.5),
                lons = np.arange(0,360,3.75),
                n_T = 100
        )

        # LINEAR MODES
        m_mixed.initiate_linear_modes()
        m_mixed.linear_modes.add_linear_modes(2)
        m_mixed.linear_modes.plot_modes()
        plt.show()

        m_mixed.linear_modes.implement_timeseries_functions()
        m_mixed.linear_modes.plot_timeseries()
        plt.show()


        # NOISE
        m_mixed.initiate_noise_modes()
        m_mixed.noise_modes.add_noise_modes()
        m_mixed.noise_modes.plot_modes()
        plt.show()

        m_mixed.noise_modes.implement_timeseries_functions()
        m_mixed.noise_modes.plot_timeseries()
        plt.show()


        # NON-LINEAR MODES
        m_mixed.initiate_non_linear_modes()
        m_mixed.non_linear_modes.add_non_linear_modes(2)
        m_mixed.non_linear_modes.plot_modes()
        plt.show()

        m_mixed.non_linear_modes.implement_timeseries_functions()
        m_mixed.non_linear_modes.plot_timeseries()
        plt.show()
        
        # NON-LINEAR CYCLIC MODES
        m_mixed.initiate_non_linear_cyclic_modes()
        m_mixed.non_linear_cyclic_modes.add_non_linear_modes(2)
        m_mixed.non_linear_cyclic_modes.plot_modes()
        plt.show()

        m_mixed.non_linear_cyclic_modes.implement_timeseries_functions()
        m_mixed.non_linear_cyclic_modes.plot_timeseries()
        plt.show()


        # TRAVERSING WAVES
        m_mixed.initiate_wave_modes()
        m_mixed.wave_modes.add_traversing_modes(2)
        m_mixed.wave_modes.plot_modes()
        plt.show()

        m_mixed.wave_modes.implement_timeseries_functions()
        m_mixed.wave_modes.plot_timeseries()
        plt.show()

        print(m_mixed.summary)

    print('MAIN COMPLETE')
