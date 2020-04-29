from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import numpy as np
import pandas as pd
import xarray as xr

import cf_units

import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

import sklearn
from scipy.stats import pearsonr
import math

def _roundup(x, base=5):
    return int(base * math.ceil(float(x)/base))
def _rounddown(x, base=5):
    return int(base * math.floor(float(x)/base))

def nino_34_index(da, three_month_averaging=True):
    """This function calculates the ONI index based on the protocol described here
    http://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
    args:
        cube: xarray of sea surface temperatures
        as_pandas: return as pandas dataframe. Else as iris cube
        averaging: apply 3 month averaging to index. Else just 1 month
    output:
        xarray
        """
    months = xr.DataArray(np.vectorize(lambda x : x.month)(da.time.values),
                  dims=['time'],
                  coords={'time':da.time})
    months.name='month'
    
    sst_clim = da.groupby(months).mean(dim='time')
    sst_anom = da.groupby(months) - sst_clim
    
    sst_anom_nino34 = sst_anom.sel(latitude=slice(5, -5), longitude=slice(190, 240))
    sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('longitude', 'latitude'))
    
    if three_month_averaging:
        oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()
        oni.name = '3_month_mean_ONI'
    else:
        oni = sst_anom_nino34_mean
        oni.name = '1_month_mean_nino34'
        
    return oni


def deviation_from_monthly_means(da):
    '''This function will allow me to change the representation of the data from absolute temperature to
    deviation from the month mean'''
    months = xr.DataArray(
            da.time.values.astype('datetime64[M]').astype(int) % 12 + 1,
            name = 'month',
            dims=['time'],
            coords={'time':da.time})
    
    da_month_means = da.groupby(months).mean(dim='time')
    da = (da.groupby(months) - da_month_means).drop('month')
    return da

def aggregate_to_year(da, deviation = True):
        years = xr.DataArray(
                da.time.values.astype('datetime64[M]').astype(int)//12 + 1970,
                name = 'year',
                dims=['time'],
                coords={'time':da.time})
        da = da.groupby(years).mean(dim='time')
        if deviation:
            da = da - da.mean(dim='year')
            
            
        da.year.values = np.vectorize(lambda x : datetime.datetime.strptime("{0:-04d}".format(x), "%Y"))(da.year.values)
        da = da.rename({"year": "time"})
        return da
    
def deviation_from_year_means(da, rolling=False):
    '''This function will allow me to change the representation of the data from absolute temperature to
    deviation from the year mean'''
    if rolling:
        da = da - da.rolling(time=12, center=True).mean()
    else:
        years = xr.DataArray(
                dates.astype('datetime64[Y]').astype(int) + 1970,
                name = 'year',
                dims=['time'],
                coords={'time':da.time})
        da_year_means = da.groupby(years).mean(dim='time')
        da = (da.groupby(years) - da_year_means).drop('year')
    return da


def signal_processing(da,
                      varname = None,
                      agg_to_year = False,
                      add_global_warming_signal = False,
                      apply_standard_scaler=True,
                      demean_all_time=False,
                      apply_area_weighting=True,
                      normalise_by_year = False,
                      normalise_by_rolling_year = False,
                      normalise_by_month=True):
    """This function applies all of the processing steps that I have developed with the option to 
    include or not include any step"""

    if agg_to_year:
        da = aggregate_to_year(da, deviation = True)

    if add_global_warming_signal:
        da = da + np.linspace(0,1, da.shape[0])[:,np.newaxis, np.newaxis]

    if not agg_to_year:

        if normalise_by_year:
            da = deviation_from_year_means(da, normalise_by_rolling_year)

        if normalise_by_month:
            da = deviation_from_monthly_means(da)
            
    if demean_all_time:
        da = da - da.mean(dim='time')

    if apply_standard_scaler:
        #ss = StandardScaler() # probably overkill
        # decide here whether you will pre-normalise or not
        da = da - da.mean(dim='time')
        da = da/da.std(dim='time')

    if apply_area_weighting:
        coslat = np.abs(np.cos(np.deg2rad(da.latitude)))**0.5
        weights = coslat #/ coslat.sum(dim=('latitude', 'longitude'))
        da = da*weights
        
    if varname is not None:
        da.name = varname
    
    return da

def model_output_to_xarray(ds, flat_components, flat_features=None, timeseries=None, 
                           model_type = 'unknown', other_model_attribs={}):
    basis_vectors = flat_components.reshape(flat_components.shape[:1]+ds.latitude.shape + ds.longitude.shape)
    model_dict = other_model_attribs
    model_dict.update({'model':model_type})
    xr_basis_vectors = xr.DataArray(basis_vectors, 
                                    name = 'model_components',
                                    attrs = model_dict,
                                coords={
                                        'component_n': np.arange(basis_vectors.shape[0]),
                                        'latitude': ds.latitude,
                                        'longitude': ds.longitude,
                                       }, 
                                dims=('component_n','latitude', 'longitude'))
    
    if timeseries is None:
        timeseries = np.matmul(flat_components, flat_features.T)
        
    xr_timeseries = xr.DataArray(timeseries,
                                 name = 'component_timeseries',
                                 coords={
                                        'component_n': np.arange(basis_vectors.shape[0]),
                                        'time': ds.time,
                                       }, 
                                dims=('component_n','time'))
    return xr.merge([xr_basis_vectors, xr_timeseries])


def component_non_orthonality(basis_vectors1, basis_vectors2):
    df =  pd.DataFrame(np.matmul(basis_vectors1, np.transpose(basis_vectors2))).round(2)
    df.index.name= 'model1'
    df.columns.name = 'model2'
    return df


def datetime_cheat_single_(dt, to_normal = True):
    '''Converts from datetime360day and back again'''
    if to_normal:
        return datetime.datetime(*dt.timetuple()[:6])
    else:
        return cf_units.cftime.Datetime360Day(*dt.timetuple()[:6])
    
    
def datetime_cheat(dt, to_normal = True):
    dtn = np.vectorize(datetime_cheat_single_)(dt, to_normal=to_normal)
    return dtn


def aggregate_pandas_to_year(df):
    df = df.reset_index()
    years = df.time.apply(lambda x: x.year)
    df = df.drop('time', axis=1)
    df = df.groupby(years).agg(np.mean)
    df.index = np.vectorize(lambda x : datetime.datetime.strptime("{0:-04d}".format(x), "%Y"))(df.index.values)
    return df
