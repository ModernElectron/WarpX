"""
Utility functions for mewarpx.
"""
import errno
import inspect
import os
import collections
import numpy as np

from pywarpx import geometry
from mewarpx import mwxconstants as constants

# http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-t
mewarpx_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

axis_labels_2d = ['r', 'x', 'z']
axis_labels_3d = ['x', 'y', 'z']
axis_dict_3d = {'x': 0, 'y': 1, 'z': 2}
axis_dict_2d = {'r': 0, 'x': 0, 'z': 1}

def init_libwarpx(ndim, rz):
    """_libwarpx requires the geometry be set before importing.

    This complicates a lot of our code if we need to delay importing it - so
    instead we import it here.

    Very Bad Things will happen if ndim and rz here are different than is
    used in the rest of the simulation!

    Arguments:
        ndim (int): Number of dimensions. Ignored for RZ.
        rz (bool): True for RZ simulations, else False.
    """
    geometry.coord_sys = 1 if rz else 0
    geometry.prob_lo = [0]*ndim
    import pywarpx._libwarpx
    # This just quiets linters like pyflakes by using the otherwise-unused
    # variable
    assert pywarpx._libwarpx


# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    """Make directory and parent directories if they don't exist.

    Do not throw error if all directories already exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def ideal_gas_density(p, T):
    """Calculate neutral gas density (in 1/cm^3) from the ideal gas law using
    pressure in Torr.

    Arguments:
        p (float): Gas pressure (Torr)
        T (float): Mean gas temperature (K)

    Returns:
        N (float): Number density of gas atoms/molecules (1/cm^3)
    """
    return (p * constants.torr_cgs) / (constants.kb_cgs * T)

def recursive_update(d, u):
    """Recursively update dictionary d with keys from u.
    If u[key] is not a dictionary, this works the same as dict.update(). If
    u[key] is a dictionary, then update the keys of that dictionary within
    d[key], rather than replacing the whole dictionary.
    """
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def return_iterable(x, depth=1):
    """Return x if x is iterable, None if x is None, [x] otherwise.
    Useful for arguments taking either a list of single value. Strings are a
    special case counted as 'not iterable'.
    Arguments:
        depth (int): This many levels must be iterable. So if you need an
            iterable of an iterable, this is 2.
    """
    if x is None:
        return None
    elif depth > 1:
        # First make sure it's iterable to one less than the required depth.
        x = return_iterable(x, depth=depth-1)
        # Now check that it's iterable to the required depth. If not, we just
        # need to nest it in one more list.
        x_flattened = x
        while depth > 1:
            if all([(isinstance(y, collections.abc.Iterable) and not isstr(y))
                    for y in x_flattened]):
                x_flattened = [z for y in x_flattened for z in y]
                depth -= 1
            else:
                return [x]
        return x

    elif isinstance(x, str):
        return [x]
    elif isinstance(x, collections.abc.Iterable):
        return x
    else:
        return [x]


def get_axis_idxs(axis1, axis2, dim=2):
    """Return the indices appropriate for the given axes and dimension.
    Arguments:
        axis1 (string): 'r', 'x', 'y' or 'z'
        axis2 (string): 'r', 'x', 'y' or 'z'
        dim (int): 2 or 3 (2D/3D)
    Returns:
        idx_list (list): [axis1_idx, axis2_idx, slice_idx, slice_str]. Here
        slice_idx is the third dimension for 3D and slice_str is its label. Both
        are None for 2D.
    """
    axes = [axis1, axis2]
    if dim not in [1, 2, 3]:
        raise ValueError("Unrecognized dimension dim = {}".format(dim))
    if dim == 1:
        return[axis_dict_2d['z'], axis_dict_2d['x'], None, None]
    for ii, axis in enumerate(axes):
        if dim == 2 and axis not in axis_labels_2d:
            raise ValueError("Unrecognized axis {} for 2D".format(axis))
        if dim == 3 and axis not in axis_labels_3d:
            if axis == 'r':
                axes[ii] = 'x'
            else:
                raise ValueError("Unrecognized axis {} for 3D".format(axis))
    if axes[0] == axes[1] or (axes[0] in ['r', 'x']
                              and axes[1] in ['r', 'x']):
        raise ValueError("axis1 and axis2 must be different")
    if dim == 2:
        return [axis_dict_2d[axes[0]], axis_dict_2d[axes[1]], None, None]
    xaxis = axis_dict_3d[axes[0]]
    yaxis = axis_dict_3d[axes[1]]
    sliceaxis = (set((0, 1, 2)) - set((xaxis, yaxis))).pop()
    s_str = (set(('x', 'y', 'z')) - set((axes[0], axes[1]))).pop()
    return [xaxis, yaxis, sliceaxis, s_str]

def get_2D_field_slice(data, xaxis, yaxis, slicevec=None, slicepos=None):
    """Return appropriate 2D field slice given the geometry.
    Arguments:
        data (np.ndarray): 2D or 3D array, depending on geometry
        xaxis (int): Index of abscissa dimension of data
        yaxis (int): Index of ordinate dimension of data
        sliceaxis (int): Index of dimension of data to slice from. None for 2D.
        slicevec (np.ndarray): 1D vector of positions along slice. None for 2D,
            or to take middle element in 3D.
        slicepos (float): Position to slice along sliceaxis (m). Default 0 if
            slicevec != None; ignored if slicevec == None.
    Returns:
        slice (np.ndarray): 2D array. Ordinate is the first dimension of the
        array, abscissa the 2nd.
    """
    data = np.array(data)
    dim = len(data.shape)
    if dim == 1:
        data = np.tile(data, (2, 1))
        # Flip x and y?
        if xaxis < yaxis:
            return data.T
        return data
    if dim == 2:
        # if slicevec is not None or slicepos is not None:
        #      logger.warning("slicevec and slicepos ignored for 2D data in "
        #                     "get_2D_field_slice()")
        # Flip x and y?
        if xaxis < yaxis:
            return data.T
        return data
    sliceaxis = (set((0, 1, 2)) - set((xaxis, yaxis))).pop()
    if slicevec is None:
        # if slicepos is not None:
        #     logger.warning("slicepos ignored when slicevec == None in "
        #                    "get_2D_field_slice()")
        idx = data.shape[sliceaxis] // 2
    else:
        if slicepos is None:
            slicepos = 0.0
        idx = np.argmin(np.abs(slicevec - slicepos))
    dslice = data.take(idx, axis=sliceaxis)
    if xaxis < yaxis:
        return dslice.T
    return dslice
