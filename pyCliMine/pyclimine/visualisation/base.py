import numpy as np

from sklearn.utils import shuffle
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import cartopy.crs as ccrs
import cmocean

import ipywidgets
import animatplot as amp


class widget_component_plot():
    
    def __init__(self, X):
        self.data = X
        self.com = 0
        self.map_proj = ccrs.PlateCarree()
        self.interactive_fig = plt.figure(figsize = (14,6))
        #self.interactive_ax = plt.axes(projection=self.map_proj)
        
        ipywidgets.interact_manual(self.update, component_number = ipywidgets.IntSlider(value=0,
                                               min=0,
                                               max=X.shape[0]-1,
                                               step=1,  continuous_update=False,))

    def update(self, component_number = 0):
        self.com = component_number
        self.interactive_fig.clear()
        self.interactive_ax = plt.axes(projections_df=self.map_proj)
        data = self.data.sel(component_n=self.com)
        p = data.plot(ax = self.interactive_ax, transform=self.map_proj)
        self.interactive_ax.coastlines()
        plt.show()


def red_noise_generator(L,s,r):
    w = np.random.normal(0,s,L)
    x = [w[0]]
    for i in range(1, L):
        x.append(r*x[i-1]+(1-r**2)**0.5*w[i])
    return np.array(x)


def sample_periodic_signal_generator(P_months, variance, n_months, redness = 0, noise_var_proportion = 0):
    t = np.arange(n_months)
    periodic_signal = np.tile(red_noise_generator(P_months, 1, 1-1e-9), n_months//P_months + 1)[:n_months]
    periodic_signal = periodic_signal/periodic_signal.std() * ((1-noise_var_proportion) * variance)**0.5
    if noise_var_proportion!=0:
        noise = red_noise_generator(len(periodic_signal), 1, redness) * (noise_var_proportion * variance)**0.5
        periodic_signal = periodic_signal + noise
    return periodic_signal


def welch_power_spectrum_plot(projections_df, components, samp_freq, enso_index=None, 
                       include_white=False, include_red=False, include_shuffled=False, 
                       include_periodics = None, periodics_keywargs = {},
                       time_axis = ('frequency', 'months'), plot_kind = 'loglog',
                       return_fig=False
                       ):
    """
    args:
        projections_df: pandas.DataFrame containing the signal.
        components: the column numbers of projections_df to evaluate.
        samp_freq: the time spacing, in months, between successive rows in projections_df.
    """
    # plot types dict
    plots_dict = {'loglog':plt.loglog, 
                  'semilogy':plt.semilogy,
                  'linear':plt.plot}
    # check inputs
    if plot_kind not in plots_dict.keys():
        raise ValueError(
            'plot_kind : {} not valid. Must be one of {}'.format(
                plot_kind,plots_dict.keys())
        )
    if time_axis[0] not in ('frequency', 'period') or \
       time_axis[1] not in ('months', 'years'):
        raise ValueError(
            'time_axis : {} not valid. Must be one of {}'.format(
                time_axis, (('frequency', 'period'), ('months', 'years')))
        )
        
    def welsh_time_axis_modifier(x1, fs, time_axis):
        '''Function completes welsh power spectrum analysis and modifes
        output for the time axis wanted'''
        f, Pxx_spec = signal.welch(x1, fs, 'flattop', 1024, scaling='spectrum')
        if time_axis[1]=='years':
            f = f*12
        if time_axis[0]=='period':
            f = f[1:]**-1
            Pxx_spec = Pxx_spec[1:]
        return f, Pxx_spec
    
    fs = samp_freq
    # signal.welch
    fig = plt.figure(figsize=(7,7))
    ############
    ax1 = plt.subplot(211)
    plt.title('Power spectrum (scipy.signal.welch)')

    for c in components:
        x1 = projections_df.iloc[:,c]
        f, Pxx_spec = welsh_time_axis_modifier(x1, fs, time_axis)
        plots_dict[plot_kind](f, np.sqrt(Pxx_spec), label=c,)

    plt.xlabel('{} [${}{}'.format(*time_axis,'^{-1}$]' if time_axis[0] == 'frequency' else '$]'))
    plt.ylabel('Linear spectrum - components')
    plt.legend(framealpha=0, loc='best')

    ###########
    if (enso_index is not None)|include_white|include_red|include_shuffled|(include_periodics is not None):
        plt.subplot(212, sharex = ax1)
        if enso_index is not None:
            f, Pxx_spec = welsh_time_axis_modifier(enso_index, fs, time_axis)
            plots_dict[plot_kind](f, np.sqrt(Pxx_spec), label='enso')

        if include_red:
            x1 = red_noise_generator(len(projections_df),projections_df.iloc[:,0].std(),0.85)
            f, Pxx_spec = welsh_time_axis_modifier(x1, fs, time_axis)
            plots_dict[plot_kind](f, np.sqrt(Pxx_spec), label='red')

        if include_white:
            x1 = np.random.normal(0,projections_df.iloc[:,0].var(),len(projections_df))
            f, Pxx_spec = welsh_time_axis_modifier(x1, fs, time_axis)
            plots_dict[plot_kind](f, np.sqrt(Pxx_spec), label='white')

        if include_shuffled:
            x1 = shuffle(projections_df.iloc[:,0].values)
            f, Pxx_spec = welsh_time_axis_modifier(x1, fs, time_axis)
            plots_dict[plot_kind](f, np.sqrt(Pxx_spec), label='shuffled 0th comp.')
            
        if include_periodics is not None:
            for P_months in include_periodics:
                x1 = sample_periodic_signal_generator(P_months, projections_df.iloc[:,0].var(), 
                                                      len(projections_df), **periodics_keywargs)
                f, Pxx_spec = welsh_time_axis_modifier(x1, fs, time_axis)
                plots_dict[plot_kind](f, np.sqrt(Pxx_spec), 
                                      label='Periodic : {} months'.format(P_months))
    
        plt.xlabel('{} [${}{}'.format(*time_axis,
                                      '^{-1}$]' if time_axis[0] == 'frequency' else '$]'))
        plt.ylabel('Linear spectrum - ONI')
        plt.legend(framealpha=0, loc='best')

    ###########
    plt.tight_layout()
    
    if return_fig:
        return fig
    plt.show()
    
    
def fft_power_spectrum_plot(projections_df, components, samp_freq, enso_index=None, 
                       include_white=False, include_red=False, include_shuffled=False,
                       return_fig=False):
    fs = samp_freq

    plt.figure()
    ############
    ax1 = plt.subplot(211)
    plt.title('Power spectrum (np.fft.fft)')

    for c in components:
        x1 = projections_df.loc[:,c]
        freqs = np.fft.fftfreq(x1.shape[0], 1/fs)
        idx = np.argsort(freqs)
        ps = np.abs(np.fft.fft(x1))**2
        plt.semilogy(freqs[idx], ps[idx], label=c)

    plt.xlabel('frequency [$month^{-1}$]')
    plt.ylabel('Linear spectrum - component 0')
    plt.legend(framealpha=0, loc='best')

    ###########

    plt.subplot(212, sharex = ax1)
    if enso_index is not None:
        x1 = enso_index
        freqs = np.fft.fftfreq(x1.shape[0], 1/fs)
        idx = np.argsort(freqs)
        ps = np.abs(np.fft.fft(x1))**2
        plt.semilogy(freqs[idx], ps[idx], label=c)

    if include_red:
        x1 = red_noise_generator(len(projections_df),projections_df.loc[:,0].std(),0.85)
        freqs = np.fft.fftfreq(x1.shape[0], 1/fs)
        idx = np.argsort(freqs)
        ps = np.abs(np.fft.fft(x1))**2
        plt.semilogy(freqs[idx], ps[idx], label='red')

    if include_white:
        x1 = np.random.normal(0,projections_df.loc[:,0].std(),len(projections_df))
        freqs = np.fft.fftfreq(x1.shape[0], 1/fs)
        idx = np.argsort(freqs)
        ps = np.abs(np.fft.fft(x1))**2
        plt.semilogy(freqs[idx], ps[idx], label='white')

    if include_shuffled:
        x1 = shuffle(projections_df.loc[:,0].values)
        freqs = np.fft.fftfreq(x1.shape[0], 1/fs)
        idx = np.argsort(freqs)
        ps = np.abs(np.fft.fft(x1))**2
        plt.semilogy(freqs[idx], ps[idx], label='shuffled 0th comp.')

    plt.xlabel('frequency [$month^{-1}$]')
    plt.ylabel('Linear spectrum - ONI')
    plt.legend(framealpha=0, loc='best')

    ###########
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    
def animate_climate_fields(n_mapseries, lats, lons, suptitle='', fps=10, 
                           save_path='', layout=None, shape=None, subtitles=[],
                           figsize=None, colorbar_clip_pct = 1,
                           share_colorbar=False):
    """Produces and shows a figure which is an inimation of n climate fields.
    Saves figure out optionally.
    Args:
        n_mapseries (numpy array):
        layout (numpy array): position of each index in grid, To leave
        position blank set the layout value to nan for that position.
        subtitles (list): subtitles for each respective element of n_mapseries"""

    N = n_mapseries.shape[0]
    
    # set plot layout
    if layout is None:
        if shape is None: shape = ((N+1)//2, 2)
        if np.product(shape)<N:
            raise ValueError('shape {} too small for {} modes'.format(layout, N))
        layout = np.arange(np.product(shape), dtype=object)
        layout[N:]=np.nan
    else:
        shape = layout.shape
    
    fig = plt.figure(figsize = figsize)
    
    ind_j, ind_i = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # flatten and ignore some positions
    layout=layout.flatten()
    mask = [~np.isnan(x) for x in layout]
    layout=layout[mask]
    ind_j = ind_j.flatten()[mask]
    ind_i = ind_i.flatten()[mask]
    
    # default subtitles
    if subtitles==[]: subtitles = ['mode {}'.format(i) for i in range(N)]
        
    if share_colorbar: 
        vamp =  np.abs(np.percentile(n_mapseries, [colorbar_clip_pct, 100-colorbar_clip_pct])).max()
    
    # spatial and time coordinates
    X,Y = np.meshgrid(lons,lats)
    t = np.arange(n_mapseries.shape[1])
    timeline = amp.Timeline(t, fps=fps)

    #create axes and set blocks
    blocks = []
    for n in range(len(layout)):
        m = int(layout[n])
        ax = plt.subplot2grid(shape,(ind_i[n], ind_j[n]))
        ax.set_aspect('equal')
        ax.set_title(subtitles[m])
        if not share_colorbar:
            vamp =  np.abs(np.percentile(n_mapseries[m], [colorbar_clip_pct, 100-colorbar_clip_pct])).max()
        block = amp.blocks.Pcolormesh(X,Y, n_mapseries[m], 
                                      ax=ax, t_axis=0,
                                      cmap=cmocean.cm.balance,
                                      vmin=-vamp, vmax=vamp)
        blocks.append(block)
    #plt.colorbar(blocks[-1].quad)

    # contruct the animation
    anim = amp.Animation(blocks, timeline)
    
    # other matplotlib modifications
    fig.suptitle(suptitle)
    plt.tight_layout()
    
    # controls
    anim.controls()
    
    # save gif
    if save_path!='':
        anim.save_gif(save_path)
        
    plt.show()