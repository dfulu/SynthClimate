import numpy as np
import os

def check_make_dir(directory):
    """Check if a directory exists and make it if it does not"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def normal_variance(projections, time_ax = 1, space_ax = (2,3)):
    sum_ax = tuple(s-1 if time_ax<s else s for s in space_ax)
    var = np.var(projections, axis=time_ax).sum(axis=sum_ax)
    return var

def area_weighted_variance(projections, lats, time_ax = 1, lat_ax = 2):
    weights = abs(np.cos(np.deg2rad(lats)))**0.5
    weights = weights.reshape(tuple(len(lats) if i==lat_ax else 1 
                                    for i in range(len(projections.shape))))
    projections=projections*weights
    D = len(projections.shape)
    space_ax = (D-2, D-1)
    var = normal_variance(projections, time_ax = time_ax, space_ax = space_ax)
    return var

def flatten_except(x, axis=(0,1)):
    axis_before = [x.shape[i] for i in np.arange(axis[0])]
    axis_before = [np.product(axis_before)] if axis_before!=[] else []
    axis_after = [x.shape[i] for i in np.arange(axis[-1]+1, len(x.shape))]
    axis_after = [np.product(axis_after)] if axis_after!=[] else []
    axis_middle = [x.shape[a] for a in axis]
    new_shape = axis_before +axis_middle+axis_after
    return  x.reshape(tuple(new_shape))


def f(ax=1):
    x = np.random.normal(0,1,30).reshape(2,3,5)
    w = np.arange(3).reshape(tuple(3 if i==ax else 1 for i in range(len(x.shape))))
    return x*w
f()