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
