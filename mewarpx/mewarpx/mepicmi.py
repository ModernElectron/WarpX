"""
Child classes of WarpX Picmi objects to support ME utility functions and
extended functionality without going through their standards.
"""
import logging

from pywarpx import picmi

# Get module-level logger
logger = logging.getLogger(__name__)


class Species(picmi.Species):

    def init(self, kw):
        """PICMI relies on C++ WarpX to translate charge & mass strings to
        floats. To get around that we have our own variables sq/sm (species
        charge/mass) that are always floats.
        """
        super(Species, self).init(kw)

        if isinstance(self.charge, str):
            if self.charge == 'q_e':
                self.sq = picmi.constants.q_e
            elif self.charge == '-q_e':
                self.sq = -picmi.constants.q_e
            else:
                raise ValueError("Unrecognized charge {}".format(self.charge))
        else:
            self.sq = self.charge

        if isinstance(self.mass, str):
            if self.mass == 'm_e':
                self.sm = picmi.constants.m_e
            elif self.mass == 'm_p':
                self.sm = picmi.constants.m_p
            else:
                raise ValueError("Unrecognized mass {}".format(self.mass))
        else:
            self.sm = self.mass