"""
Placeholder for assembly implementations.
"""

from pywarpx import picmi

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

    def __init__(self, center, height, radius, direction,
                 has_fluid_inside, V, T, WF, name):

        """Basic initialization.

        Arguments:
            center_x (string): The coordinates of the center of the cylinder. In the
                format of "x y z" for 3D or "x y" or "x z" for 2D. Coordinates are in (m)
            height (float): The full height (or length) of the cylinder (m)
            radius (float): The radius of the cylinder (m)
            direction (int): The axis that the cylinder is aligned to. If this EB is in 3D
                then 0=x, 1=y, 2=z. If working in 2D as XY then 0=x, 1=y. If working in 2D
                as XZ then 0=x,1=Z.
            has_fluid_inside (boolean): Whether or not the EB object should be considered
                an object in free space or as a hole in an infinite conductor.
            V (float): Voltage (V)
            T (float): Temperature (K)
            WF (float): Work function (eV)
            name (str): Assembly name
        """
        super(Cylinder, self).__init__(V=V, T=T, WF=WF, name=name)

        self.center = center
        self.height = height
        self.radius = radius
        self.direction = direction
        self.has_fluid_inside = has_fluid_inside

    def get_picmi_object(self):
        return picmi.EmbeddedBoundary(
            geom_type="cylinder", cylinder_center=self.center,
            cylinder_radius=self.radius, cylinder_height=self.height,
            cylinder_direction=self.direction, has_fluid_inside=self.has_fluid_inside,
            potential=self.V
        )
