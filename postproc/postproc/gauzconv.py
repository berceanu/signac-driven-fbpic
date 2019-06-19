''' Module for Gaussian convolution. '''

import numpy as np

# signal processing, needed for convolution
from scipy import signal


def gauss_kern_nd(n, sizes):
    r"""Constructs a Gaussian kernel in any number of dimensions.
    
    Parameters
    ----------
    n : int
        The number of dimensions. Must be 1 or above.
    sizes : list
        The sizes along the various dimensions, ie. in 3D this would be [size_x, size_y, size_x]. The length of
        this list should be either 1 or n. If it only contains one element, eg. [size], it is assumed that
        size_x = size_y = size_x = size.
    
    Returns
    -------
    g : ndarray
        The kernel, with dimensions (in 3d) (2*size_x+1, 2*size_y+1, 2*size_z+1).
    """
    assert n > 0, 'at least 1d required'
    no_sizes = len(sizes)
    assert no_sizes == 1 or no_sizes == n, 'either give one size or all of them'
    
    for i in range(n - no_sizes):
        sizes.append(sizes[0])
        
    slices = tuple(slice(-size,size+1,None) for size in sizes)
    
    XXX = np.mgrid[slices]
    
    g = np.ones(XXX.shape[1:], dtype=np.float64)
    
    for X, size in zip(XXX, sizes):
        g = g * np.exp(-X**2/size)
        
    return g / g.sum()


def smooth(data, n, sizes):
    r"""Smoothens the input by performing a convolution with a Gaussian kernel.
    
    Parameters
    ----------
    data : ndarray
        Input data, with ``n`` dimensions.
    n : int
        The number of dimensions. Must be 1 or above.
    sizes : list
        The sizes along the various dimensions, ie. in 3D this would be [size_x, size_y, size_x]. The length of
        this list should be either 1 or n. If it only contains one element, eg. [size], it is assumed that
        size_x = size_y = size_x = size.
        
    Returns
    -------
    out : ndarray
        The smoothed input, of shape ```data.shape - 2 * sizes```.
    """
    
    g = gauss_kern_nd(n, sizes)
    out = signal.convolve(data, g, mode='valid')
    return out
