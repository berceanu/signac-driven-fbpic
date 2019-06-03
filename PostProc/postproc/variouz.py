''' Module containing utility functions that don't fit anywhere else. '''

import math
import numpy as np

from collections import OrderedDict


def center_linspace(space):
    r"""
    
    Parameters
    ----------
    space : ndarray
        A one-dimensional array, of linearly spaced numbers. 
    
    Returns
    -------
    out : ndarray
        Also an array of the same spacing, this time centered around 0.
    """
    
    assert np.ndim(space) == 1
    size = space.size
    first = space[0]
    last = space[-1]
    length = last - first
    end = length/2
    
    _, step = np.linspace(first, last, size, retstep=True)
    
    out, out_step = np.linspace(-end, end, size, retstep=True) 
    assert math.isclose(out_step, step)
    return out


def get_sdf_files(path):
    r"""Given a ``path``, it returns an ordered dictionary containing all the .sdf files in that path, in the form {#### : /../../####.sdf}"""
    
    sdfs = glob.glob(os.path.join(path, '*.sdf')) 
    sdf_dict = {int(sdf.split('.')[0][-4:]):sdf for sdf in sdfs}
    sdf_ord_dict = OrderedDict(sorted(sdf_dict.items(), key=lambda t: t[1])) 
    
    return sdf_ord_dict
