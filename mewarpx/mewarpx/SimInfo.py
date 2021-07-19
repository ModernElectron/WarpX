import numpy as np

class SimInfo(object):

    """Store basic simulation parameters used throughout analysis.
    This must contain:
        SimInfo.nxyz (int nx, int ny, int nz)
        SimInfo.pos_lims (floats xmin, xmax, ymin, ymax, zmin, zmax)
        SimInfo.geom (str 'XZ', 'RZ' or 'XYZ')
        SimInfo.dt (float)
        SimInfo.periodic (bool)
    Note:
        This base class has been created to provide additional plotting
        functionality, and allow post-run creation of this object if needed.
    """

    def __init__(self, nxyz, pos_lims, geom, dt, periodic=True):
        self.nxyz = nxyz
        self.pos_lims = pos_lims
        self.geom = geom
        self.dt = dt
        self.periodic = periodic

    def get_vec(self, axis):
        if axis == 'r':
            raise ValueError("RZ plotting is not implemented yet.")
            return self.get_rvec()
        axis_dict = {0: 0, 1: 1, 2: 2, 'x': 0, 'y': 1, 'z': 2}
        axis = axis_dict[axis]
        npts = self.nxyz[axis]
        xmin = self.pos_lims[2*axis]
        xmax = self.pos_lims[2*axis + 1]
        # There is one more point on the grid than cell number
        return np.linspace(xmin, xmax, npts + 1)
