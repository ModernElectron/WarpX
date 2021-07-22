"""Class for installing a field diagnostic with optional plotting"""

from mewarpx.mwxrun import mwxrun
from mewarpx.diags_store.diag_base import WarpXDiagnostic

from pywarpx import callbacks, picmi
from mewarpx import plotting

class FieldDiagnostic(WarpXDiagnostic):
    def __init__(self, diag_steps, diag_data_list, grid, name, write_dir,
                 plot_on_diag_step=False, plot_data_list=None,**kwargs):
        """
        Arguments:
            diag_steps (int): Run the diagnostic with this period.
            data_list (list (str)): A list of criteria to collect
                and print diagnostic information for.
            plot_on_diag_step (bool): Whether or not to plot the diagnostic data
                on each step.

        """
        self.diag_steps = diag_steps
        self.diag_data_list = diag_data_list
        self.grid = grid
        self.name = name
        self.write_dir = write_dir
        self.plot_on_diag_step = plot_on_diag_step
        self.plot_data_list = plot_data_list

        super(FieldDiagnostic, self).__init__(diag_steps, **kwargs)

        self.add_field_diag()

        if self.plot_on_diag_step:
            assert self.plot_data_list is not None, "plot_on_diag_step was True but plot_data_list is None!"
            self.add_plot_callback()

    def add_field_diag(self):
        diagnostic = picmi.FieldDiagnostic(
            grid=self.grid,
            period=self.diag_steps,
            data_list=self.diag_data_list,
            name=self.name,
            write_dir=self.write_dir
        )

        mwxrun.simulation.add_diagnostic(diagnostic)

    def add_plot_callback(self):
        callbacks.installafterstep(self.plot)

    def plot(self):
        if self.check_timestep():
            # the number of times a diagnostic has been run
            current_diag_num = int(mwxrun.get_it() / self.diag_steps)
            plotting.plot_parameters(self.plot_data_list, "after_diag_step", current_diag_num)
