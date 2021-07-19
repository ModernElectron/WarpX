import numpy as np

"""
Placeholder for assembly implementations.
"""


class Assembly(object):

    """An assembly represents any shape in the simulation; usually a conductor.

    While V, T, and WF are required, specific use cases may allow None to be
    used for these fields.
    """

    def __init__(self, V, T, WF, name):
        """Basic initialization.

        Arguments:
            V (float): Voltage (V)
            T (float): Temperature (K)
            WF (float): Work function (eV)
            name (str): Assembly name
        """
        self.V = V
        self.T = T
        self.WF = WF
        self.name = name

    def getvoltage(self):
        """Allows for time-dependent implementations to override this."""
        return self.V


class ZPlane(Assembly):

    """A semi-infinite plane."""

    def __init__(self, z, zsign, V, T, WF, name):
        """Basic initialization.

        Arguments:
            z (float): The edge of the semi-infinite plane (m)
            zsign (int): =1 to extend from z to +inf, or =-1 to extend from
                -inf to z.
            V (float): Voltage (V)
            T (float): Temperature (K)
            WF (float): Work function (eV)
            name (str): Assembly name
        """
        super(ZPlane, self).__init__(V=V, T=T, WF=WF, name=name)

        self.z = z

        self.zsign = int(round(zsign))
        if self.zsign not in [-1, 1]:
            raise ValueError("self.zsign = {} is not either -1 or 1.".format(
                self.zsign))


class Cylinder(Assembly):
    """A finite Cylinder """
    def __init__(self, center_x, center_z, radius, V, T, WF, name):
        """Basic initialization.
        Arguments:
            center_x (float): The x-coordinates of the center of the cylinder.
                Coordinates are in (m)
            center_z (float): The z-coordinates of the center of the cylinder.
                Coordinates are in (m)
            radius (float): The radius of the cylinder (m)
            V (float): Voltage (V)
            T (float): Temperature (K)
            WF (float): Work function (eV)
            name (str): Assembly name
        """
        super(Cylinder, self).__init__(V=V, T=T, WF=WF, name=name)
        # Y is always treated as the height
        self.center_x = center_x
        self.center_z = center_z
        self.height = 1 # m
        self.radius = radius

    def isinside(self, X, Y, Z, aura):
        """
        Calculates which grid tiles are within the cylinder.
        Arguments:
            X (np.ndarray): array of x coordinates of flattened grid.
            Y (np.ndarray): array of y coordinates of flattened grid.
            Z (np.ndarray): array of z coordinates of flattened grid.
            aura (float): extra space around the conductor that is considered inside. Useful
                for small, thin conductors that don't overlap any grid points. In
                units of meters.
        Returns:
            result (np.ndarray): array of flattened grid where all tiles
                inside the cylinder are 1, and other tiles are 0.
        """
        result = np.where((X - self.center_x)**2 + (Z - self.center_z)**2 <= (self.radius + aura)**2,
                             1, 0)

        return result

    def calculatenormal(self, px, py, pz):
        """
        Calculates Normal of particle inside/outside of conductor to nearest
        surface of conductor
        Arguments:
            px (np.ndarray): x-coordinate of particle in meters.
            py (np.ndarray): y-coordinate of particle in meters.
            pz (np.ndarray): z-coordinate of particle in meters.
        Returns:
            nhat (np.ndarray): Normal of particles of conductor to nearest
                surface of conductor.
        """
        # distance from center of cylinder
        dist = np.sqrt((px - self.center_x)**2 + (pz - self.center_z)**2)
        nhat = np.zeros([3, len(px)])
        nhat[0, :] = (px - self.center_x) / dist
        nhat[2, :] = (pz - self.center_z) / dist
        return nhat

    # TODO: Below is partially complete code for calculating normals of an arbitrary
    # emitter, adapted from warp fortran code
    # def calculatenormal(self, px, py, pz):
    #     """
    #     Calculates Normal of particle inside/outside of conductor to nearest
    #     surface of conductor
    #     Arguments:
    #         px (np.ndarray): x-coordinate of particle in meters.
    #         py (np.ndarray): y-coordinate of particle in meters.
    #         pz (np.ndarray): z-coordinate of particle in meters.
    #     Returns:
    #         nhat (np.ndarray): Normal of particles of conductor to nearest
    #             surface of conductor.
    #     """
    #     noc = len(px)

    #     # Indices of particles that are inside of conductor
    #     inside_idx = np.where(self.isinside(px,py,pz))

    #     # Holds indices of nearest grid
    #     ix = np.zeros(px.shape)
    #     iy = np.zeros(py.shape)
    #     iz = np.zeros(pz.shape)

    #     # Format of deltas
    #     # Negative Values indicate point inside of conductor
    #     #dels[0,:]  ==> distance from point to conductor edge along - x-hat
    #     #dels[1,:]  ==> distance from point to conductor edge along + x-hat
    #     #dels[2,:]  ==> distance from point to conductor edge along - y-hat
    #     #dels[3,:]  ==> distance from point to conductor edge along + y-hat
    #     #dels[4,:]  ==> distance from point to conductor edge along - z-hat
    #     #dels[5,:]  ==> distance from point to conductor edge along + z-hat

    #     # Get deltas to conductor
    #     dels = np.absolute(self.griddistance(ix, px, py, pz).dels)

    #     # Unpack array and invert deltas
    #     mindelx = np.minimum(dels[0,:],dels[1,:])
    #     negidx = np.where(dels[0,:] < dels[1,:])
    #     mindelx[negidx] *= -1.

    #     mindely = np.minimum(dels[2,:],dels[3,:])
    #     negidy = np.where(dels[2,:] < dels[3,:])
    #     mindely[negidy] *= -1.

    #     mindelz = np.minimum(dels[4,:],dels[5,:])
    #     negidz = np.where(dels[4,:] < dels[5,:])
    #     mindelz[negidz] *= -1.

    #     # Initialize inverse deltas to largepos
    #     inf = float("inf")
    #     idepsx = inf*np.ones(mindelx.shape)
    #     idepsy = inf*np.ones(mindely.shape)
    #     idepsz = inf*np.ones(mindelz.shape)

    #     # Find non-zero indicies
    #     nzidxx = np.compress(mindelx != 0., np.arange(idepsx.size))
    #     nzidxy = np.compress(mindely != 0., np.arange(idepsy.size))
    #     nzidxz = np.compress(mindelz != 0., np.arange(idepsz.size))

    #     # Avoids division by zero
    #     idepsx[nzidxx] = 1./mindelx[nzidxx]
    #     idepsy[nzidxy] = 1./mindely[nzidxy]
    #     idepsz[nzidxz] = 1./mindelz[nzidxz]

    #     # Calculate normal vector
    #     D = 1./np.sqrt(idepsx**2 + idepsy**2 + idepsz**2)
    #     nhat = np.zeros([3,noc])
    #     nhat[0,:] = -D*idepsx
    #     nhat[1,:] = -D*idepsy
    #     nhat[2,:] = -D*idepsz

    #     # This is correct normal orientation for test point outside of conductor
    #     # That is is points outward
    #     # Invert the normal if the test point sits inside of conductor
    #     if len(inside_idx) > 0:
    #         nhat[:,inside_idx] *= -1.

    #     return nhat


    # def getdeltas(ix, px, py, pz):

    #     dels = np.zeros((6, len(ix)))
    #     delmx = dels[0, :]
    #     delpx = dels[1, :]
    #     delmy = dels[2, :]
    #     delpy = dels[3, :]
    #     delmz = dels[5, :]
    #     delpz = dels[5, :]
    #     fuzz = 1.e-13

    #     ctheta = px / np.sqrt(px**2 + pz**2)
    #     stheta = pz / np.sqrt(px**2 + pz**2)
    #     cphi = np.sqrt(px**2 + pz**2) / np.sqrt(px**2 + pz**2 + py**2)
    #     sphi = py / np.sqrt(px**2 + pz**2 + py**2)

    #     for i in range(len(ix)):
