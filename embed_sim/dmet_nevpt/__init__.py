from pyscf.lib.numpy_helper import NPArrayWithTag
from pyscf import lib
import numpy as np

def tag_array(arr, **kwargs):
    '''Attach attributes to numpy ndarray. The attribute name and value are
    obtained from the keyword arguments.
    '''
    # Make a shadow copy in any circumstance by converting it to an nparray.
    # If a is an object of NPArrayWithTag, all attributes will be lost in this
    # conversion. They need to be restored.
    t = np.asarray(arr).view(NPArrayWithTag)

    if isinstance(arr, NPArrayWithTag):
        t.__dict__.update(arr.__dict__)
    t.__dict__.update(kwargs)
    return t

lib.tag_array = tag_array

from . import nevpt
def NEVPT(mc, state=0, canonstep=1, overlap_threshold=1e-8, classes=None, verbose=None):
    return nevpt.NEVPT(mc, state=state, canonstep=canonstep, overlap_threshold=overlap_threshold, classes=classes, verbose=verbose)

NEVPT2 = FIC_NEVPT = FIC_NEVPT2 = NEVPT