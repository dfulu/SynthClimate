import abc

import cf_units
import datetime
import os
import json
import time
import six

import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cmocean

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib as mpl
import seaborn as sns

from pyclimine.generate import mixed_modes_master
from pyclimine.process import signal_processing
from pyclimine.visualisation import animate_climate_fields
from pyclimine.utils import area_weighted_variance, normal_variance, flatten_except

#######################
# UTILITY FUNCTIONS
#######################

def parse_docstring_for_plot_name(fn, i=0):
    docstring = fn.__doc__
    try:
        namestring = re.sub(' +', ' ',
           re.search(r'Name:([^:]*)', docstring).group(1)
          ).replace('\n', '')
    except:
        namestring = 'model_specific_plot_{}'.format(i)

    return namestring

def nandot(x,y):
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        x_mask = np.isnan(x)
        y_mask = np.isnan(y)
        x[x_mask]=0
        y[y_mask]=0
    return np.dot(x,y)

def single_comparison_metric(D, D_):
            N = D.shape[0]
            d = D - D.mean(axis=0)
            d_= D_ - D_.mean(axis=0)
            sigma_s  = (1./N * np.nansum(D*D) 
                       - nandot(np.mean(D, axis=0), np.mean(D, axis=0)))**0.5
            sigma_s_ = (1./N * np.nansum(D_*D_)
                       - nandot(np.mean(D_, axis=0), np.mean(D_, axis=0)))**0.5
            m = 1./N * np.nansum(d * d_)/(sigma_s*sigma_s_)
            return m
        
def one_hot_column(x, n):
            y = np.zeros(x.shape)
            y[:,n]=x[:,n]
            return y
    
def all_to_all_comparison_metric(D, D_):
            m_mat = np.zeros((D.shape[0], D_.shape[0]))
            for i in range(D.shape[0]):
                for j in range(D_.shape[0]):
                    m_mat[i,j]=single_comparison_metric(D[i], D_[j])
            return m_mat
        
def _is_group_nan(group):
    return np.any(np.isnan(np.asarray(group).astype(float)))

#######################
# BASE TEST CLASS
#######################

class model_test_object(object, metaclass=abc.ABCMeta):
    
    def __init__(self, data_object, output_directory, overwrite=False):
        """Base class init method"""
        if not isinstance(data_object, mixed_modes_master):
            raise ValueError('Unsupported data object. Data object must be instance of `mixed_modes_master`')
        self.data_object=data_object
        self.output_directory = output_directory
        self.directory_setup(overwrite)
        self.data_modes = self.data_object.n_modes
        
    # OVERWRITABLE METHODS
    @abc.abstractmethod
    def instantiate_model(self, **kwargs):
        """Abstract method which will instantiate a model object
        and initiate it as self.model"""
        
    @property
    @abc.abstractmethod
    def model_modes(self):
        """Abstract property of how mnay modes the model will have"""
    
    @abc.abstractmethod
    def train_model(self, **kwargs):
        """Abstract method which will train the model"""
        
    @abc.abstractmethod
    def retrieve_model_modes(self):
        """Abstract method which will retireve the model modes
        from the model"""
        
    @abc.abstractmethod
    def model_project_modes(self):
        """Abstract method which will project the model modes
        over the data series"""
        
    @abc.abstractmethod
    def model_flatten_modes(self):
        """Abstract method which will project and flatten the 
        model modes over the data series"""
        
    @abc.abstractmethod
    def model_mode_snaps(self):
        """Abstract method which returns a linear estimate/caricature of the modes.
        This will just be used for plotting comparisons between linear and 
        non-linear modes"""
        
    @abc.abstractmethod
    def save_model(self):
        """Abstract method which will save the model to file"""
    
    @property
    @abc.abstractmethod
    def model_summary(self):
        """Abstract property which will generate model summary as
        as pandas dataframe"""
    
    @staticmethod
    def _extra_processing(data):
        """Function to apply any model-specific extra processing to 
        the data"""
        return data
    
    def extra_plot_example(self):
        """Method returns pyplot figure without needing any arguments
        to be fed in. 
        Name:example_extra_plot:"""        
        x = np.linspace(0,4*np.pi, 1000)
        y = np.sin(x)
        fig = plt.figure(figsize=(8,8))
        plt.plot(x,y)
        plt.title('this data object has {} modes'.format(self.data_object.n_modes))
        return fig
    
    ##################################
    # SET METHODS
    ##################################
    
    # DIRECTORIES
    
    @staticmethod
    def check_make_dir(directory):
        """Check if a directory exists and make it if it does not"""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def directory_setup(self, overwrite):
        if os.path.exists(self.output_directory) and not overwrite:
            raise ValueError('Directory already exists. Overwrite set to False')
        self.check_make_dir(self.output_directory)
    
    # DATA PROCESSING
    
    def process_data(self):
        """Method to retrieve and assign the projected modes and the 
        flattened data"""
        projected_modes = self.data_object.project_to_xarray()
        self.projected_modes_shape = projected_modes.shape
        self.processed_projected_modes = self.preprocess(
            projected_modes, keep_axis=(0,1))
        self.X_train = self.processed_projected_modes.sum(axis=0)
            
    def preprocess(self, data, keep_axis=(0,)):
        """Method takes array of data and pre-processes 
        the data ready for mdoelling."""
        data = signal_processing(data,
                          agg_to_year = False,
                          add_global_warming_signal = False,
                          demean_all_time=True,
                          apply_standard_scaler=False,
                          apply_area_weighting=True,
                          normalise_by_year = False,
                          normalise_by_rolling_year = False,
                          normalise_by_month=False)
        data = self._extra_processing(data)
        data = flatten_except(data.values, axis=keep_axis)
        return data
    
    # METRIC TEST
    
    def mode_match_metric(self):
        """Apply projection matching metric"""
        data_projections = self.processed_projected_modes
        model_projections = flatten_except(self.model_projections, axis=(0,1))
        metric_matrix = all_to_all_comparison_metric(data_projections, model_projections)
        return metric_matrix
    
    def data_similarity_metric(self):
        data_projections = self.processed_projected_modes
        metric_matrix = all_to_all_comparison_metric(data_projections, data_projections)
        return metric_matrix
        
    def calculate_overall_metric(self):
        df = self.test_summary[self.test_summary.data_mode_type!='noise'].dropna()
        return (df.match_metric*df.data_variance).sum()/df.data_variance.sum()
    
    def calculate_mean_match_metric(self):
        df = self.test_summary[self.test_summary.data_mode_type!='noise'].dropna()
        return df.match_metric.mean()
        
    @staticmethod
    def _calculate_groupings(similarity_array):
        sim = similarity_array.copy()
        N,M = sim.shape
        groupings = np.zeros((max(N,M), 2), dtype=object)
        i=0
        # find best matches
        while i<min(N,M):
            ind = np.unravel_index(np.argmax(sim, axis=None), sim.shape)
            groupings[i]=ind
            sim[ind[0],:] = -np.inf
            sim[:,ind[1]] = -np.inf
            i+=1
        # append non-matches
        if N!=M:
            axis = np.argmax([N,M])
            other_axis = np.argmin([N,M])
            groupings.T[axis,i:] = np.setdiff1d(np.arange(max(N,M)), groupings.T[axis,:i])
            groupings.T[other_axis,i:] = np.nan
        return groupings
    
    def test_model(self):
        self.model_projections = self.model_project_modes()
        self.match_metric_matrix = self.mode_match_metric()
        self.groupings = self._calculate_groupings(self.match_metric_matrix)
        
        self.data_similarity_matrix = self.data_similarity_metric()
        
        self.data_variances = normal_variance(self.processed_projected_modes, space_ax=(2,))
        self.model_variances = normal_variance(self.model_projections, space_ax=(2,))
        
        self.overall_metric = self.calculate_overall_metric()
        self.mean_match_metric = self.calculate_mean_match_metric()
        
        self.data_modes_variance = normal_variance(
            self.processed_projected_modes[self.data_object.summary.index[
                     self.data_object.summary.mode_type!='noise'].values]
                 .sum(axis=0), time_ax=0, space_ax=(1,)
        )
        
        self.model_modes_variance = normal_variance(
            self.model_projections[self.test_summary.dropna().model_index.values.astype(int)]
                 .sum(axis=0), time_ax=0, space_ax=(1,)
        )
        
        self.data_total_variance = normal_variance(self.X_train, time_ax=0, space_ax=(1,))
        self.model_total_variance = float(normal_variance(self.model_projections.sum(axis=0), 
                                                    time_ax=0, space_ax=(1,)))
        
        self.reconstruction_rmsne = np.mean(
            ((self.X_train - self.model_flatten_modes())/self.X_train)**2
        )**0.5
        return
    
    # TEST SUMMARY
         
    @property
    def test_summary(self):
        if not hasattr(self, 'model_variances'):
            raise NameError('must run `.test_model` method before summary available')
        # match matric
        match_metric = [self.match_metric_matrix[tuple(g)] 
                        if not _is_group_nan(g) else np.nan 
                        for g in self.groupings]
        # data variances
        data_variances = [self.data_variances[g[0]] 
                          if not _is_group_nan(g[0]) else np.nan 
                          for g in self.groupings]
        # model variances
        model_variances = [self.model_variances[g[1]] 
                          if not _is_group_nan(g[1]) else np.nan 
                          for g in self.groupings]
        # mode type
        data_mode_type = [self.data_object.summary.mode_type.values[g[0]] 
                          if not _is_group_nan(g[0]) else np.nan 
                          for g in self.groupings]
        
        df = pd.DataFrame(
                {
                'data_mode_type': data_mode_type,
                'data_index': self.groupings[:,0],
                'model_index': self.groupings[:,1],
                'match_metric': match_metric,
                'data_variance': data_variances,
                'model_variance': model_variances,
                }
            )
        return df
    
    def save_test_summary(self):
        test_summary = {'overall_score':self.overall_metric, 
                        'mean_match_metric':self.mean_match_metric,
                        'reconstruction_rmsne': self.reconstruction_rmsne,
                        'data_total_variance':self.data_total_variance,
                        'model_total_variance':self.model_total_variance,
                        'data_modes_variance':self.data_modes_variance,
                        'model_modes_variance':self.data_modes_variance,
                        'data_model_match_matrix':self.match_metric_matrix.tolist(),
                        'data_similarity_matrix':self.data_similarity_matrix.tolist(),
                        'mode_df':self.test_summary.to_json()
                       }
        data_summary = {'text_note':self.data_object.text_note,
                        'mode_df':self.data_object.summary.to_json()}
        with open(os.path.join(self.output_directory,'test_summary.json'), 'w') as fp:
            json.dump(test_summary, fp, sort_keys=False, indent=4)
        with open(os.path.join(self.output_directory,'data_summary.json'), 'w') as fp:
            json.dump(data_summary, fp, sort_keys=False, indent=4)
        return

    # CORE PLOTS
    
    def core_plot_spatial_variance(self):
        """Name:data_variance_map:"""
        y = signal_processing(self.data_object.project_to_xarray().sum(axis=0),
                              agg_to_year = False,
                              add_global_warming_signal = False,
                              apply_standard_scaler=False,
                              apply_area_weighting=False,
                              normalise_by_year = False,
                              normalise_by_rolling_year = False,
                              normalise_by_month=False)

        # make plot of variances
        y_std = y.std(dim='time')
        map_proj = ccrs.PlateCarree()

        fig = plt.figure(figsize = (12,5))
        ax = plt.axes(projection=map_proj);
        p = y_std.plot(ax = ax, transform=map_proj)
        ax.coastlines()
        return fig
    
    def core_plot_mode_snaps(self, invert_match=True):
        """Name:mode_snaps:"""
        # model linear caricatures
        data_modes = self.data_object.mode_snaps()
        model_modes = self.model_mode_snaps()

        fig = plt.figure(figsize = (20, 5*self.model_modes))
        map_proj = ccrs.PlateCarree()
        
        n_plot_rows = len(self.test_summary)
        
        for i,row in self.test_summary.iterrows():

            if not np.isnan(row.data_index):
                ax = plt.subplot(n_plot_rows, 2, 2*i+1, projection=map_proj)
                z = data_modes[row.data_index]
                v_abs_max = abs(z).max()
                xz = xr.DataArray(z, 
                                  coords = [self.data_object.lats, 
                                            self.data_object.lons-180], 
                                  dims=['latitude', 'longitude'])
                xz.plot(ax = ax, transform=map_proj, 
                                cmap=cmocean.cm.balance, 
                                vmin=-v_abs_max, vmax=v_abs_max)
                ax.coastlines()
                ax.set_title("data mode {} (mode type: {})".format(
                                row.data_index, row.data_mode_type)
                                 )

            if not np.isnan(row.model_index):
                ax = plt.subplot(n_plot_rows, 2, 2*i+2, projection=map_proj)
                z = model_modes[row.model_index]
                invert = invert_match and (~np.isnan(row.data_index)) and \
                    ((z*data_modes[row.data_index]).sum()<0)
                z = -z if invert else z
                v_abs_max = abs(z).max()
                xz = xr.DataArray(z, 
                                  coords = [self.data_object.lats, 
                                            self.data_object.lons-180], 
                                  dims=['latitude', 'longitude'])
                xz.plot(ax = ax, transform=map_proj, 
                                cmap=cmocean.cm.balance, 
                                vmin=-v_abs_max, vmax=v_abs_max)
                ax.coastlines()
                ax.set_title(
                    "model mode {} (match metric: {:.3f} | inverted: {})"
                    .format(row.model_index, row.match_metric, invert)
                )

        plt.tight_layout()
        return fig
        
    def save_model_plots(self, extra=False):
        """Method which runs either the core plotting functions
        or the extra model-specific plotting functions and saves 
        the figure they produce in the correct place"""
        i=0
        if extra:
            startstring = 'extra_plot'
            directory = os.path.join(self.output_directory, 'plots', 'extra_plots')
        else:
            startstring = 'core_plot'
            directory = os.path.join(self.output_directory, 'plots')
        
        self.check_make_dir(directory)
        
        for name in dir(self):
            if (name.startswith(startstring))&(name!='extra_plot_example'):
                i+=1
                method = getattr(self, name)
                fig = method()
                filename = parse_docstring_for_plot_name(method, i=i)
                plt.savefig(os.path.join(directory, filename+'.png'))
                plt.close('all')
                
                
    # ANIMATION
    
    def animate_matches(self, save_path='', 
                        match_indexes=[], figsize=None,
                        t_ind_start=0, t_ind_stop=100):
        model_projections=self.model_projections.reshape(
            self.model_projections.shape[:2]+self.projected_modes_shape[2:]
        )
        data_projections=self.processed_projected_modes.reshape(
            self.projected_modes_shape
        )
        # concatenate data
        data = np.concatenate(
            [data_projections, model_projections], 
            axis=0)[:,t_ind_start:t_ind_stop,...]
        
        # create informative subtitles
        data_mode_types =  self.data_object.summary.mode_type
        match_metrics = self.test_summary[['model_index','match_metric']]\
                            .set_index('model_index').sort_index().match_metric
        

        subtitles = ["data mode {} (mode type: {})".format(i, data_mode_types[i]) 
                     for i in range(self.data_modes)] +\
                    ["model mode {} (match metric: {:.3f})".format(i, match_metrics[i]) 
                     for i in range(self.model_modes)]
        
        # set layout
        layout_df = self.test_summary[['data_index','model_index']]
        layout_df.model_index = layout_df.model_index.apply(
            lambda x: x + self.data_modes if x!=np.nan else x)
        layout = layout_df.values.astype(float)
        if match_indexes!=[]: layout=layout[list(match_indexes)]
        # create animation
        animate_climate_fields(data, 
                               self.data_object.lats, 
                               self.data_object.lons, 
                               suptitle='', fps=2, 
                               save_path=save_path, 
                               layout=layout, 
                               subtitles=subtitles, 
                               figsize=figsize)
        
    def save_model_projections(self):
        """Save the projections of each model mode to .nc file. The projections saved
        are comparable to `processed_projected_modes` attribute."""
        path = os.path.join(self.output_directory, 'model_projected_modes.nc')
        ps = self.model_projections
        ps = ps.reshape(ps.shape[:2]+self.projected_modes_shape[2:])
        modes = np.arange(ps.shape[0])
        t = np.arange(ps.shape[1])
        data = xr.DataArray(ps, 
                     coords=[modes, t, self.data_object.lats, self.data_object.lons], 
                     dims=['mode', 'time', 'latitude', 'longitude'])
        data.to_netcdf(path, mode='w')
        
    def save_data_projections(self):
        """Save the projections of each processed data mode to .nc file. 
        The projections saved are comparable to `model_projections` attribute."""
        path = os.path.join(self.output_directory, 'data_processed_projected_modes.nc')
        ps = self.processed_projected_modes.reshape(
            self.projected_modes_shape
        )
        modes = np.arange(ps.shape[0])
        t = np.arange(ps.shape[1])
        data = xr.DataArray(ps, 
                     coords=[modes, t, self.data_object.lats, self.data_object.lons], 
                     dims=['mode', 'time', 'latitude', 'longitude'])
        data.to_netcdf(path, mode='w')
        
    
    # MODEL SPECIFIC SAVING WRAPPERS
    
    def save_model_summary(self):
        return
        

#######################
# SPECIFIC MODEL TESTS
#######################
    


class pca_test_object(model_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components
    
        # model specific imports
        from sklearn.decomposition import PCA
        
        self.model = PCA(n_components = n_components, **kwargs)
        
    @property
    def model_modes(self):
        return self._model_modes

    def train_model(self):
        self.model_series = self.model.fit_transform(self.X_train)
        
    def retrieve_model_modes(self):
        return self.model.components_
    
    def model_project_modes(self):
        return self.model_series.T[..., np.newaxis] * self.model.components_[:, np.newaxis, :]
        
    def model_flatten_modes(self):
        return np.matmul(self.model_series, self.model.components_)
    
    def model_mode_snaps(self):
        return self.model.components_.reshape((self.model_modes, 
                                               self.data_object.lats.shape[0], 
                                               self.data_object.lons.shape[0]))
        
    def save_model(self):
        print('save model not implemented')
    
    def model_summary(self):
        print('model summary not implemented')

        
########################################################
        


class nmf_test_object(pca_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components
    
        # model specific imports
        from sklearn.decomposition import NMF
        
        self.model = NMF(n_components = n_components, **kwargs)
        
    @staticmethod
    def _extra_processing(data):
        """Function to apply any model-specific extra processing to 
        the data"""
        return data - data.min()
    
########################################################

class fa_test_object(pca_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components

        # model specific imports
        from sklearn.decomposition import FactorAnalysis
        
        self.model = FactorAnalysis(n_components = n_components, 
                                   svd_method = 'randomized')

########################################################

class ica_test_object(pca_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components

        # model specific imports
        from sklearn.decomposition import FastICA
        
        self.model = FastICA(n_components = n_components, 
                                max_iter=200, tol=0.0001)

########################################################

class kpca_test_object(pca_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components
    
        # model specific imports
        from sklearn.decomposition import KernelPCA
        
        self.model = KernelPCA(n_components = n_components, 
                               kernel='poly',
                               fit_inverse_transform=True)
        
    def retrieve_model_modes(self):
        raise NotImplementedError('retrieve_model_modes')
    
    def model_project_modes(self):
        return np.array([self.model.inverse_transform(one_hot_column(self.model_series, n)) 
                         for n in range(self.model_modes)])
        
    def model_flatten_modes(self):
        return self.model.inverse_transform(self.model_series)
    
    def model_mode_snaps(self):
        mm = np.stack([self.model_series.min(axis=0), self.model_series.max(axis=0)])
        proj_mm = np.array([self.model.inverse_transform(one_hot_column(mm, n)) 
                         for n in range(self.model_modes)])
        return (proj_mm[:,1] - proj_mm[:,0]).reshape((self.model_modes, 
                                               self.data_object.lats.shape[0], 
                                               self.data_object.lons.shape[0]))

########################################################

class sfa_test_object(model_test_object):
    
    def instantiate_model(self, n_components, pca_components=None, **kwargs):
        if pca_components is None: pca_components=2*n_components
        self._model_modes = n_components
    
        # model specific imports
        from mdp.nodes import SFANode, PCANode
        
        self.model = PCANode(output_dim=pca_components) + SFANode(output_dim=n_components, **kwargs)
        
    @property
    def model_modes(self):
        return self._model_modes

    def train_model(self, epochs=2):
        for i in range(epochs):
            print('Training epoch {} of {}'.format(i, epochs))
            self.model.train(self.X_train)
        self.model_series = self.model.execute(self.X_train)
        
    def retrieve_model_modes(self):
        
        def one_hot(n,N):
            x = np.zeros(N)
            x[n]=1
            return x
        
        return self.model.inverse(np.array([one_hot(n,self.model_modes) for n in range(self.model_modes)]))
    
    def model_project_modes(self):
        return np.array([self.model.inverse(one_hot_column(self.model_series, n)) 
                         for n in range(self.model_modes)])
        
    def model_flatten_modes(self):
        return self.model.inverse(self.model_series)
    
    def model_mode_snaps(self):
        return self.retrieve_model_modes().reshape((self.model_modes, 
                                               self.data_object.lats.shape[0], 
                                               self.data_object.lons.shape[0]))
        
    def save_model(self):
        print('save model not implemented')
    
    def model_summary(self):
        print('model summary not implemented')
        
########################################################

class dmd_test_object(model_test_object):

    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components
        
        # model specific imports
        from pydmd import DMD

        self.model = DMD(svd_rank=n_components, opt=False, exact=False, **kwargs)
        
    @property
    def model_modes(self):
        return self._model_modes

    def train_model(self):
        self.model.fit(self.X_train.T)
        # calculate latent variable time series
        self.model_series = np.zeros((self.X_train.shape[0], self.model_modes))
        for i in range(self.X_train.shape[0]):
            self.model_series[i] = self.model._compute_amplitudes(self.model.modes, 
                                                     self.X_train[i:i+1].T, 
                                                     self.model.eigs, 
                                                     False)
        
    def retrieve_model_modes(self):
        return self.model.modes.T.real
        
    def model_project_modes(self):
        return np.real(self.model.modes.T[:,np.newaxis,:] * self.model_series.T[:,:,np.newaxis])
    
    def model_flatten_modes(self):
        return np.real(np.matmul(self.model.modes, self.model_series.T).T)
    
    def model_mode_snaps(self):
        return self.retrieve_model_modes().reshape((self.model_modes, 
                                               self.data_object.lats.shape[0], 
                                               self.data_object.lons.shape[0]))
        
    def save_model(self):
        return
    
    def model_summary(self):
        return
    
    def extra_plot_dynamics(self):
        """Name:dynamics:"""
        fig = plt.figure(figsize=(8,8))
        for dynamic in self.model.dynamics:
            plt.plot(dynamic.real)
            plt.title('Dynamics')
            plt.xlabel('t')
        return fig

    def extra_plot_eig_circle(self):
        """Name:dynamics_eigenvalues_circle:"""
        fig = plt.figure(figsize=(8,8))
        self.model.plot_eigs(show_axes=True, show_unit_circle=True)
        return fig

    def extra_plot_render_mpl_table(self, col_width=3.0, row_height=0.625, font_size=14,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0, **kwargs):
        """Name:eignvalue_table:"""
        
        model = self.model
        
        data=pd.Series(model.eigs, name='eigenvalues').apply(lambda x: '{:.3f}'.format(x)).to_frame()
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        return fig
    
########################################################

class baseline_test_object(model_test_object):
    
    def instantiate_model(self, n_components, **kwargs):
        self._model_modes = n_components      
        
        # model specific imports
        from pyclimine.generate import linear_modes_object
        
        self.model = linear_modes_object(
                    lats = self.data_object.lats,
                    lons = self.data_object.lons,
                    n_T = self.data_object.n_T)
        self.model.add_linear_modes(n_components, **kwargs)
        
        self.model.components_ = flatten_except(self.model.modes, axis=(0,))
        self.model.components_ =self.model.components_ / np.diag(np.matmul(
            self.model.components_, self.model.components_.T)
            )[:,np.newaxis]**0.5
        
    @property
    def model_modes(self):
        return self._model_modes

    def train_model(self):
        self.model_series = np.matmul(self.X_train, self.model.components_.T)
        
    def retrieve_model_modes(self):
        return self.model.components_
    
    def model_project_modes(self):
        return self.model_series.T[..., np.newaxis] * self.model.components_[:, np.newaxis, :]
        
    def model_flatten_modes(self):
        return np.matmul(self.model_series, self.model.components_)
    
    def model_mode_snaps(self):
        return self.model.components_.reshape(self.model.modes.shape)
        
    def save_model(self):
        print('save model not implemented')
    
    def model_summary(self):
        print('model summary not implemented')
        
########################################################

class autoencoder_test_object(model_test_object):
    
    def instantiate_model(self, n_components, lr=2e3, decay=1e-6, momentum=0.2, 
                          **kwargs):
        self._model_modes = n_components    
        # model specific imports 
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras import metrics
        from keras.optimizers import SGD

        # layers
        input_img= Input(shape=(self.X_train.shape[1],))
        encoded = Dense(n_components, activation='linear')(input_img)
        decoded = Dense(self.X_train.shape[1], activation='linear')(encoded)

        # autoencoder
        autoencoder=Model(input_img, decoded)

        # encoder
        encoder = Model(input_img, encoded)

        # decoder
        encoded_input = Input(shape=(n_components,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # compile autoencoder
        sgd = SGD(lr=lr, decay=decay, momentum=momentum)
        autoencoder.compile(optimizer=sgd, 
                            loss='mean_squared_error', #else kullback_leibler_divergence
                            metrics=[metrics.mae,])
        
        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
    @property
    def model_modes(self):
        return self._model_modes

    def train_model(self, epochs=200, batch_size=200):
        # train autoencoder
        self.model.fit(self.X_train, self.X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
        )
        self.model_series = self.encoder.predict(self.X_train)
        
    def retrieve_model_modes(self):
        return self.model.get_weights()[-2]
    
    def model_project_modes(self):
        return np.array([self.decoder.predict(one_hot_column(self.model_series, n)) 
                         for n in range(self.model_modes)])
        
    def model_flatten_modes(self):
        return self.model.predict(self.X_train)
    
    def model_mode_snaps(self):
        return self.retrieve_model_modes().reshape(self.model_modes, 
                                               self.data_object.lats.shape[0], 
                                               self.data_object.lons.shape[0])
        
    def save_model(self):
        print('save model not implemented')
    
    def model_summary(self):
        print('model summary not implemented')
        
########################################################

if __name__ == '__main__':
    
    import tempfile
    generate_data=True
    add_linear=True
    add_noise=False
    add_non_linear=False
    add_wave=False
    generate_plots=True
    # output file
    directory  = os.path.join(tempfile.gettempdir(), 'testdir')# 
    if generate_data:
        print('GENERATING DATA')
        # LINEAR
        data_object = mixed_modes_master(
                    lats = np.arange(90,-91,-2.5),
                    lons = np.arange(0,360,3.75),
                    n_T = 1000)
        
        if add_linear:
            data_object.initiate_linear_modes()
            data_object.linear_modes.add_linear_modes(3)
            data_object.linear_modes\
                .implement_timeseries_functions(vrs=[1])
            
        if add_noise:
            # NOISE
            data_object.initiate_noise_modes()
            data_object.noise_modes.add_noise_modes()
            data_object.noise_modes.implement_timeseries_functions(vr=1.5)
        
        if add_non_linear:
            # NON-LINEAR MODES
            data_object.initiate_non_linear_modes()
            data_object.non_linear_modes.add_non_linear_modes(1)
            data_object.non_linear_modes.implement_timeseries_functions(vrs=[0.85])

        if add_wave:
            # TRAVERSING WAVES
            data_object.initiate_wave_modes()
            data_object.wave_modes.add_traversing_modes(1)
            data_object.wave_modes.implement_timeseries_functions(vrs=[1.2])
    
    print('RUNNING TEST')
    # instantiate test and process data
    print('- initiating test')
    test = dmd_test_object(data_object, directory, overwrite=True)
    print('- processing data')
    test.process_data()
    # instantiate and train model on data
    print('- initiating model')
    test.instantiate_model(n_components=3)
    print('- training model')
    test.train_model()
    # run the metric test on the model
    print('- running test_model')
    test.test_model()
    print('- saving test summary')
    test.save_test_summary()
    if generate_plots:
        # create and save the base plots
        print('- generating and saving core plots')
        test.save_model_plots(extra=False)
        # create and save the model specific plots
        print('- generating and saving other plots')
        test.save_model_plots(extra=True)
    # generate and save the model and summary
    print('- saving model')
    #test.save_model()
    print('- saving model summary')
    #test.save_model_summary()
    print(test.test_summary)

    if False:
        test.animate_matches(save_path='', match_indexes=[], figsize=(8,33),
                             t_ind_start=0, t_ind_stop=100)

    print('DONE')
    


