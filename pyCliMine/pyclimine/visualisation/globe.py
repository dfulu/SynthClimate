import plotly.plotly as py
import numpy as np           
from scipy.io import netcdf  
import warnings
from numpy import pi, sin, cos
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import os
import cmocean


def cmocean_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

balance = cmocean_to_plotly(cmocean.cm.balance, 100)

def degree2radians(degree):
    #convert degrees to radians
    return degree*pi/180

def mapping_map_to_sphere(lon, lat, radius=1):
    #this function maps the points of coords (lon, lat) to points onto the  sphere of radius radius
    lon=np.array(lon, dtype=np.float64)
    lat=np.array(lat, dtype=np.float64)
    lon=degree2radians(lon)
    lat=degree2radians(lat)
    xs=radius*cos(lon)*cos(lat)
    ys=radius*sin(lon)*cos(lat)
    zs=radius*sin(lat)
    return xs, ys, zs


def planet_plotly(lat, lon, z, zero_meaned=False, cb_title='', plot_title=''):
    lat = lat[::-1]     # invert the latitude vector -> South to North
    olr = z # olr= outgoing longwave radiation

    # Shift 'lon' from [0,360] to [-180,180]
    tmp_lon = np.array([lon[n]-360 if l>=180 else lon[n] 
                       for n,l in enumerate(lon)])  # => [0,180]U[-180,2.5]

    i_east, = np.where(tmp_lon>=0)  # indices of east lon
    i_west, = np.where(tmp_lon<0)   # indices of west lon
    lon = np.hstack((tmp_lon[i_west], tmp_lon[i_east]))  # stack the 2 halves

    # Correspondingly, shift the olr array
    olr_ground = np.array(olr)
    olr = np.hstack((olr_ground[:,i_west], olr_ground[:,i_east]))

    # Get list of of coastline, country, and state lon/lat 
    country_traces_path = os.path.join(os.path.dirname(__file__), 'country_coastline_traces.npz')
    traces_file = np.load(country_traces_path)
    country_lons = traces_file['country_lons'].tolist()
    country_lats = traces_file['country_lats'].tolist()
    cc_lats = traces_file['coastline_lats'].tolist()
    cc_lons = traces_file['coastline_lons'].tolist()

    #concatenate the lon/lat for coastlines and country boundaries:
    lons=cc_lons+[None]+country_lons
    lats=cc_lats+[None]+country_lats

    xs, ys, zs=mapping_map_to_sphere(lons, lats, radius=1.01)# here the radius is slightly greater than 1 
                                                             #to ensure lines visibility; otherwise (with radius=1)
                                                             # some lines are hidden by contours colors

    boundaries=dict(type='scatter3d',
                   x=xs,
                   y=ys,
                   z=zs,
                   mode='lines',
                   line=dict(color='black', width=1)
                  )

    clons=np.array(lon.tolist()+[180], dtype=np.float64)
    clats=np.array(lat, dtype=np.float64)
    clons, clats=np.meshgrid(clons, clats)

    XS, YS, ZS=mapping_map_to_sphere(clons, clats)

    nrows, ncolumns=clons.shape
    OLR=np.zeros(clons.shape, dtype=np.float64)
    OLR[:, :ncolumns-1]=np.copy(np.array(olr,  dtype=np.float64))
    OLR[:, ncolumns-1]=np.copy(olr[:, 0])
    cmin = -max([abs(olr.min()), olr.max()]) if zero_meaned else olr.min()
    cmax =  max([abs(olr.min()), olr.max()]) if zero_meaned else olr.max()

    text=np.asarray([['lon: {:.2f} <br>lat: {:.2f} <br>z: {:.2f}'.format(clons[i,j], clats[i, j], OLR[i][j]) 
                      for j in range(ncolumns)] for i in range(nrows)])

    sphere=dict(type='surface',
                x=XS, 
                y=YS, 
                z=ZS,
                colorscale=balance,
                surfacecolor=OLR,
                cmin=cmin, 
                cmax=cmax,
                colorbar=dict(thickness=20, len=0.75, ticklen=4, title= cb_title),
                text=text,
                hoverinfo='text')

    noaxis=dict(showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                ticks='',
                title='',
                zeroline=False)

    layout3d=dict(title=plot_title,
                  font=dict(family='Balto', size=14),
                  width=800, 
                  height=800,
                  scene=dict(xaxis=noaxis, 
                             yaxis=noaxis, 
                             zaxis=noaxis,
                             aspectratio=dict(x=1,
                                              y=1,
                                              z=1),
                             camera=dict(eye=dict(x=1.15, 
                                         y=1.15, 
                                         z=1.15)
                                        )
                ),
                paper_bgcolor='rgba(235,235,235, 0.5)'  
               )

    fig=dict(data=[sphere, boundaries], layout=layout3d)
    return fig
