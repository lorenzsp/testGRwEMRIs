#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# utils for mcmc.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from eryn.moves.gaussian import reflect_cosines_array
import corner

try:
    import cupy as xp
    # set GPU device
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False


def draw_initial_points(mu, cov, size, intrinsic_only=False):    
    
    tmp = np.random.multivariate_normal(mu, cov, size=size)
    
    # for ii in range(tmp.shape[0]):
    #     Rot = special_ortho_group.rvs(tmp.shape[1])
    #     tmp[ii] = np.random.multivariate_normal(mu, (Rot.T @ cov @ Rot))
    
    if intrinsic_only:
        for el in [-2,-3]:
            tmp[:,el] = tmp[:,el]%(2*np.pi)
    else:
        # ensure prior
        for el in [10,11]:
            tmp[:,el] = tmp[:,el]%(2*np.pi)
        
        tmp[:,6],tmp[:,7] = reflect_cosines_array(tmp[:,6],tmp[:,7])
        tmp[:,8],tmp[:,9] = reflect_cosines_array(tmp[:,8],tmp[:,9])
        
        # tmp[:,6] = np.cos(np.random.uniform(0.,2*np.pi,size=len(tmp[:,6])))
        # tmp[:,7] = np.random.uniform(0.,2*np.pi,size=len(tmp[:,7]))
        # tmp[:,8] = np.cos(np.random.uniform(0.,2*np.pi,size=len(tmp[:,8])))
        # tmp[:,9] = np.random.uniform(0.,2*np.pi,size=len(tmp[:,9]))
    
    return tmp

def spectrogram(x, window_size=4*256, step_size=64, fs=1/10):
    # Calculate number of time steps
    n_timesteps = (len(x) - window_size) // step_size + 1
    
    # Initialize spectrogram array
    spectrogram = xp.zeros((window_size // 2 + 1, n_timesteps))
    
    # Compute spectrogram
    for t in range(n_timesteps):
        # Extract windowed segment
        segment = x[t * step_size:t * step_size + window_size]
        
        # Apply window function (Hann window)
        windowed_segment = segment * xp.hanning(window_size)
        
        # Compute FFT
        fft_result = xp.fft.fft(windowed_segment)
        
        # Store magnitude spectrum
        spectrogram[:, t] = xp.abs(fft_result[:window_size // 2 + 1])
    
    # Calculate time array
    time_array = np.arange(0, len(x) / fs, step_size / fs)
    
    # Calculate frequency array
    frequency_array = np.fft.fftfreq(window_size, 1 / fs)[:window_size // 2 + 1]
    
    return spectrogram.get(), time_array, frequency_array  # Transfer data from GPU to CPU

def get_spectrogram(h,dt,name):
    # Compute spectrogram
    spec, time_array, frequency_array = spectrogram(h, fs=1/dt)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(spec), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (days)')
    plt.ylabel('Frequency (Hz)')
    newt = np.arange(0, time_array.max() / 3600 / 24, 100, dtype=int)
    xtick_loc = np.interp(newt, time_array / 3600 / 24, np.arange(len(time_array)))
    plt.xticks(xtick_loc, newt)
    newf = np.arange(0., 0.05, 0.01)
    ytick_loc = np.interp(newf, np.abs(frequency_array), np.arange(len(frequency_array)))
    plt.yticks(ytick_loc, newf)
    # plt.ylim(0,np.interp(1e-3, np.abs(frequency_array), np.arange(len(frequency_array))))
    plt.savefig(name)

def pad_to_next_power_of_2(arr):
    original_length = len(arr)
    next_power_of_2 = int(2 ** xp.ceil(np.log2(original_length)))

    # Calculate the amount of padding needed
    pad_length = next_power_of_2 - original_length

    # Pad the array with zeros
    padded_arr = xp.pad(arr, (0, pad_length), mode='constant')

    return padded_arr

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def get_plot_sky_location(qK,phiK,qS,phiS,name=None):
    # draw the SSB frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->')

    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='k')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='k')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='k',label='SSB')
    ax.add_artist(a)

    ax.text(1.1, 0, 0, r'$x$')
    ax.text(0, 1.1, 0, r'$y$')
    ax.text(0, 0, 1.1, r'$z$')

    # sky direction
    th, ph, lab = qS, phiS, 'Sky location'
    x_ = np.sin(th) * np.cos(ph)
    y_ = np.sin(th) * np.sin(ph)
    z_ = np.cos(th)
    a = Arrow3D([0, x_], [0, y_], [0, z_], **arrow_prop_dict, color='blue', label=lab)
    ax.add_artist(a)
    ax.scatter(x_,y_,z_,s=40,label='source')

    # sky spin
    th, ph, lab = qK, phiK, 'MBH Spin'
    x_s = np.sin(th) * np.cos(ph)
    y_s = np.sin(th) * np.sin(ph)
    z_s = np.cos(th)
    a = Arrow3D([x_, x_+x_s], [y_, y_+y_s], [z_, z_+z_s], **arrow_prop_dict, color='red', label=lab)
    ax.add_artist(a)

    ax.view_init(azim=-70, elev=20)
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.legend()
    if name is None:
        plt.savefig('skylocalization.pdf')
    else:
        plt.savefig(name)


from scipy.interpolate import CubicSpline
S_git = np.genfromtxt('./LISA_Alloc_Sh.txt')
Sh_X = CubicSpline(S_git[:,0], S_git[:,1])

def get_sensitivity_stas(f, **kwargs):
    """
    Calculate the LISA Sensitivity curve as defined in https://arxiv.org/abs/2108.01167.
    
    arguments:
        f (double scalar or 1D np.ndarray): Frequency array in Hz

    returns:
        1D array or scalar: S(f) with dimensions of seconds.

    """
    return Sh_X(np.abs(f))
    
    
def get_fft(sig1, dt):
    return xp.fft.fft(xp.asarray(sig1),axis=1)*dt

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None, title=None, ylim=None, weights=None,):
    """
    Plots multiple corners on top of each other.

    Parameters:
    - samples_list: list of numpy arrays
        List of MCMC samples for each corner plot.
    - sample_labels: list of strings
        List of labels for each set of samples.
    - name_save: string, optional
        Name of the file to save the plot. If not provided, the plot will be displayed.
    - corn_kw: dict, optional
        Additional keyword arguments to pass to the corner.corner function.
    - title: string, optional
        Title for the plot.
    - ylim: tuple, optional
        The y-axis limits for the marginalized corners.

    Returns:
    - None (if name_save is not provided) or saves the plot as a PDF file.

    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('Set1',)
    colors = [cmap(i) for i in range(n)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define the plot range for each dimension
    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )

    # Update corner plot keyword arguments
    corn_kw = corn_kw or {}
    corn_kw.update(range=plot_range)
    list_maxy = []
    
    if weights is None:
        weights = [get_normalisation_weight(len(samples_list[idx]), max_len) for idx in range(0, n)]
    else:
        weights = [get_normalisation_weight(len(samples_list[idx]), max_len)*weights[idx] for idx in range(0, n)]
    # Create the first corner plot
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=weights[0],
        **corn_kw
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))
    maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
    # append maxy
    list_maxy.append(maxy)
    
    # Overlay the remaining corner plots
    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=weights[idx],
            color=colors[idx],
            **corn_kw
        )
        axes = np.array(fig.axes).reshape((ndim, ndim))
        maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
        # append maxy
        list_maxy.append(maxy)
    list_maxy =np.asarray(list_maxy)


    # Set y-axis limits for the marginalized corners
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_ylim((0.0,np.max(list_maxy,axis=0)[i]))

    # Add legend
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=35,
        frameon=False,
        bbox_to_anchor=(0.5, ndim+1),
        loc="upper right",
        title=title,
        title_fontsize=35,
    )

    # Adjust plot layout
    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=None, hspace=0.15)


    # Save or display the plot
    if name_save is not None:
        plt.savefig(name_save+".pdf", pad_inches=0.2, bbox_inches='tight')
    else:
        plt.show()
