import numpy as np
from typing import Optional

class Randman:
    """ Randman (numpy version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, seed, embedding_dim, manifold_dim, alpha=2, use_bias=False, prec=1e-3, max_f_cutoff=1000):
        """ Initializes a randman object.
        
        Args
        ----
        embedding_dim : The embedding space dimension
        manifold_dim : The manifold dimension
        alpha : The power spectrum fall-off exponenent. Determines the smoothness of the manifold (default 2)
        use_bias: If True, manifolds are placed at random offset coordinates within a [0,1] simplex.
        prec: The precision parameter to determine the maximum frequency cutoff (default 1e-3)
        """
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.use_bias = use_bias
        self.dim_embedding = embedding_dim
        self.dim_manifold = manifold_dim
        self.f_cutoff = int(np.min((np.ceil(np.power(prec,-1/self.alpha)),max_f_cutoff)))
        self.spect = 1.0/((np.arange(self.f_cutoff)+1)**self.alpha)
        self.params_per_1d_fun = 3
        self.init_random()
        
           
    def init_random(self):
        self.params = self.rng.uniform(low=0, high=1, size=(self.dim_embedding, self.dim_manifold, self.params_per_1d_fun, self.f_cutoff))
        if not self.use_bias:
            self.params[:,:,0,0] = 0
        
    def eval_random_function_1d(self, x, theta):
        nfreq = np.arange(self.spect.shape[0])[...,None]
        theta = theta[...,None]
        return np.sum(theta[0]*self.spect[...,None]*np.sin( 2*np.pi*(nfreq*x[None,...]*theta[1] + theta[2]) ), axis=0)

    def eval_random_function(self, x, params):
        tmp = np.ones(len(x))
        for d in range(self.dim_manifold):
            tmp *= self.eval_random_function_1d(x[:,d], params[d])
        return tmp
    
    def eval_manifold(self, x):
        dims = []
        for i in range(self.dim_embedding):
            dims.append(self.eval_random_function(x, self.params[i]))
        data = np.stack( dims, axis=0 ).T
        return data
    
    def get_random_manifold_samples(self, nb_samples):
        x = self.rng.uniform(0, 1, size=(nb_samples,self.dim_manifold))
        y = self.eval_manifold(x)
        return x,y


def standardize(x: np.ndarray, eps=1e-7):
    mi = x.min(axis=0)
    ma = x.max(axis=0)
    return (x-mi)/(ma-mi+eps)


def make_spiking_dataset(rng: np.random.Generator, nb_classes=10, nb_units=100, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True):
    """ Generates event-based generalized spiking randman classification/regression dataset. 
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work. 
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args: 
        rng: A numpy random generator
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
    Returns: 
        A tuple of data,labels. The data is structured as numpy array 
        (sample x event x 2 ) where the last dimension contains 
        the relative [0,1] (time,unit) coordinates and labels.
    """
  
    data = []
    labels = []

    max_value = np.iinfo(np.int64).max
    randman_seeds = rng.integers(max_value, size=(nb_classes,nb_spikes) )

    for k in range(nb_classes):
        x = rng.random((nb_samples,dim_manifold))
        submans = [ Randman(randman_seeds[k,i], nb_units, dim_manifold, alpha=alpha) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(np.repeat(np.arange(nb_units).reshape(1,-1), nb_samples, axis=0))
            times.append(y)

        units = np.concatenate(units,axis=1)
        times = np.concatenate(times,axis=1)
        events = np.stack([times,units],axis=2)
        data.append(events)
        labels.append(k*np.ones(len(units)))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0).astype(np.int32)

    if shuffle:
        idx = np.arange(len(data))
        rng.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

    data[:,:,0] *= nb_steps*step_frac
    # data = np.array(data, dtype=int)

    return data, labels


def convert_spike_times_to_raster(spike_times: np.ndarray, timestep: float = 1.0, max_time: Optional[float] = None, num_neurons: Optional[int] = None, dtype=None):
    """
    Convert spike times array to spike raster array. 
    For now, all neurons need to have same number of spike times.
    
    Args:
        spike_times: MoreArrays, spiketimes as array of shape (batch_dim x spikes/neuron X 2)
            with final dim: (times, neuron_id)
    """

    if dtype is None:
        dtype = np.int16
    # spike_times = spike_times.astype(np.uint16)
    if num_neurons is None:
        num_neurons = int(np.nanmax(spike_times[:,:,1]))+1
    if max_time is None:
        max_time = np.nanmax(spike_times[:,:,0])
    num_bins = int(max_time / timestep + 1)

    spike_raster = np.zeros((spike_times.shape[0], num_bins, num_neurons), dtype=np.float32)
    batch_id = np.arange(spike_times.shape[0]).repeat(spike_times.shape[1])
    spike_times_flat = (spike_times[:, :, 0].flatten() / timestep).astype(dtype)
    neuron_ids = spike_times[:, :, 1].flatten().astype(dtype)
    np.add.at(spike_raster, (batch_id, spike_times_flat, neuron_ids), 1)
    return spike_raster

def make_spike_raster_dataset(rng, nb_classes=10, nb_units=100, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True):
    """ Generates event-based generalized spiking randman classification dataset. 
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work. 
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args: 
        rng: A numpy random generator
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
    Returns: 
        A tuple of data,labels. The data is structured as numpy array 
        (sample x event x 2 ) where the last dimension contains 
        the relative [0,1] (time,unit) coordinates and labels.
    """
    spike_times, labels = make_spiking_dataset(rng, nb_classes, nb_units, nb_steps, step_frac, dim_manifold, nb_spikes, nb_samples, alpha, shuffle)
    spike_raster = convert_spike_times_to_raster(spike_times)
    return spike_raster, labels