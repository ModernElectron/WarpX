"""Test full 1D diode run with diagnostics."""
# [[[TODO]]] NOT TOUCHED, JUST COPIED FROM WARP, SO FAR
from __future__ import division

from builtins import range
import os
import sys

import pytest
import numpy as np
import pandas

# Necessary to load Warp as library (because Warp processes arguments even when
# loaded as a module!!!)
del sys.argv[1:]

import warp
from warp.data_dumping.openpmd_diag import ParticleDiagnostic

from metools import analysis, diags
from metools import init_restart_util, runtools, util, warputil
from metools.setups_store import diode_setup


@pytest.mark.parametrize(
    ("name"),
    [
        'Run1D_alldiags_2DScraper',
        'Run1D_alldiags_1DScraper',
    ]
)
def test_run1D_alldiags(capsys, name):
    basename = "Run1D_alldiags"
    # Include a random run number to allow parallel runs to not collide. Using
    # python randint prevents collisions due to numpy rseed below
    init_restart_util.initialize_testingdir(name)

    # Initialize each run with consistent, randomly-chosen, rseed. Use a random
    # seed instead for initial dataframe generation.
    # np.random.seed()
    np.random.seed(92160881)

    # Histograms only work with 2D scraper at the moment so we test each
    # combination
    use_2d_scraper = "2DScraper" in name

    # Specific numbers match older run for consistency
    run = diode_setup.DiodeRun_V1(
        CATHODE_TEMP=1473.15,
        CATHODE_A=3.0e5,
        CATHODE_PHI=2.11,
        ANODE_TEMP=400,
        ANODE_PHI=1.2,
        V_ANODE_CATHODE=1.21,
        D_CA=4.0001e-05,
        NPARTPERSTEP=200,
        TOTAL_CROSSINGS=2.0,
        DIAG_CROSSINGS=2.0,
        J_TOLERANCE=0.001,
        CFL_FACTOR=2.0,
        OFFSET=1e-06,
        FORCE_2D_SCRAPER=use_2d_scraper,
        MERGING=True,
        MERGING_DV=100000,
        MERGING_PERPERIOD=20,
        MERGING_DXFAC=2,
        MERGING_XYFAC=10,
        CHECK_CHARGE_CONSERVATION=False,
    )

    run.setup_run(
        init_reflection=True,
        init_merging=True,
        init_traceparticles=True,
        init_runinfo=True,
        init_fluxdiag=True,
        init_resultsinfo=True
    )

    diags.TextDiag(run.diag_steps // 10, preset_string='perfdebug')

    # Particle diagnostic directory
    part_diag_dir = 'diags/xzsolver'
    particle_diagnostic = ParticleDiagnostic(
        period=run.diag_steps, top=warp.top, w3d=warp.w3d,
        species={species.name: species for species in warp.listofallspecies},
        comm_world=warp.comm_world, lparallel_output=False,
        write_dir=part_diag_dir,
        particle_data=["position", "momentum", "weighting"])
    warp.installafterstep(particle_diagnostic.write)

    runtools.FieldsDiag(solver=run.solver, diag_steps=run.diag_steps,
                        plot=True, process_phi=True,
                        process_E=True, process_rho=True,
                        process_rhob=False, barrier_slices=[0e-6],
                        max_dim=6, dpi=200
                        )

    if use_2d_scraper:
        runtools.PHistDiag(
            quantity_spec=[
                ('uxbirth', -run.setupinfo.vmax/4., run.setupinfo.vmax/4.),
                ('uzbirth', 0, run.setupinfo.vmax/4.),
            ],
            scraper=run.scraper,
            diag_steps=run.diag_steps,
            linres=30,
            jslist=0,
        )

        runtools.PHistDiag(
            quantity_spec=[
                ('zold', warp.w3d.zmmin, warp.w3d.zmmax),
            ],
            scraper=run.scraper,
            diag_steps=run.diag_steps,
            linres=50,
            jslist=0,
            name='particle_histogramcurrent'
        )

    warputil.warp_generate()

    # Run the main WARP loop
    while not run.runresults.terminate_flag:
        warp.step(run.diag_steps)

    #######################################################################
    # Cleanup and final output                                            #
    #######################################################################

    run.runresults.finalize_save()

    out, _ = capsys.readouterr()

    filelist = [
        "diags/fields/Barrier_index_0000000000.pdf",
        "diags/fields/Barrier_index_0000000000.png",
        "diags/fields/Barrier_index_0000000106.pdf",
        "diags/fields/Barrier_index_0000000106.png",
        "diags/fields/Electric_field_strength_0000000000.npy",
        "diags/fields/Electric_field_strength_0000000000.pdf",
        "diags/fields/Electric_field_strength_0000000000.png",
        "diags/fields/Electric_field_strength_0000000106.npy",
        "diags/fields/Electric_field_strength_0000000106.pdf",
        "diags/fields/Electric_field_strength_0000000106.png",
        "diags/fields/Electrostatic_potential_0000000000.npy",
        "diags/fields/Electrostatic_potential_0000000000.pdf",
        "diags/fields/Electrostatic_potential_0000000000.png",
        "diags/fields/Electrostatic_potential_0000000106.npy",
        "diags/fields/Electrostatic_potential_0000000106.pdf",
        "diags/fields/Electrostatic_potential_0000000106.png",
        "diags/fields/Net_charge_density_0000000000.npy",
        "diags/fields/Net_charge_density_0000000106.npy",
        "diags/fields/Net_charge_density_0000000106.pdf",
        "diags/fields/Net_charge_density_0000000106.png",
        "diags/traces/trace_0000000106.npz",
        "diags/xzsolver/hdf5/data0000000106.h5",
        "diags/fluxes/anode_plane_scraped.csv",
        "diags/fluxes/cathode_scraped.csv",
        "diags/results.dpkl",
        "diags/results.txt",
        "diags/runinfo.dpkl",
    ]

    if use_2d_scraper:
        filelist += [
            "diags/histograms/particle_histogram_0000000106.npy",
            "diags/histograms/particle_histogramcurrent_0000000106.npy",
            "diags/histograms/particle_histogramcurrent_setup.p",
            "diags/histograms/particle_histogramcurrent_zold_0000000106.pdf",
            "diags/histograms/particle_histogram_setup.p",
            "diags/histograms/particle_histogram_uxbirth_0000000106.pdf",
            "diags/histograms/particle_histogram_uxbirth_uzbirth_0000000106.pdf",
            "diags/histograms/particle_histogram_uzbirth_0000000106.pdf",
        ]

    print(out)
    assert "Step: 106; Diagnostic period: 1" in out

    for filename in filelist:
        assert os.path.isfile(filename)

    myrun = analysis.RunProcess(".", use_minerva=False)
    myrun.print_analysis()
    outtext, _ = capsys.readouterr()
    print(outtext)

    # Gather results to check.  The index call here ensures there's one row to
    # assign into in the new DataFrame.
    rdict = myrun.get_analysis()
    df = pandas.DataFrame(index=list(range(1)))
    df['cathode_J_emit'] = rdict['cathode_all']['emit_J_full']
    df['cathode_J_abs'] = rdict['cathode_all']['abs_J_full']
    df['anode_J_abs'] = rdict['anode_all']['abs_J_full']

    assert util.test_df_vs_ref(basename, df)
