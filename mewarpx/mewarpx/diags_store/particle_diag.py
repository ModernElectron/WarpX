"""Diagnostic code that wraps the picmi.ParticleDiagonstics class"""
import logging
import time
from mewarpx.diags_store.diag_base import WarpXDiagnostic

import numpy as np

from pywarpx import callbacks, picmi

from mewarpx.mwxrun import mwxrun

logger = logging.getLogger(__name__)

class WarpXParticleDiag(WarpXDiagnostic):

    def __init__(self, diag_steps, species=None, data_list=None, **kwargs):
        self.species = species
        self.data_list = data_list

        if self.species is None:
            self.species = mwxrun.simulation.species
        if self.data_list is None:
            self.data_list = ["position", "momentum", "weighting"]

        super(WarpXParticleDiag, self).__init__(diag_steps=diag_steps, **kwargs)
        self.add_particle_diag()

    def add_particle_diag(self):
        particle_diag = picmi.ParticleDiagnostic(
            name="p_diag",
            period=self.diag_steps,
            species=self.species,
            data_list=self.data_list
        )

        mwxrun.simulation.add_diagnostic(particle_diag)

    def post_processing():
        raise NotImplementedError