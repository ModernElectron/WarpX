import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy

from mewarpx import util


class ArrayPlot(object):

    """Initialize and plot the passed array data. Many processing steps."""

    param_defaults = {
        "points": None,
        "numpoints": 7,
        "sweepaxlabel": None,
        "xlabel": None,
        "ylabel": None,
        "title": None,
        "titlestr": "Array plot",
        "titleunits": "",
        "titleline2": None,
        'scale': 'symlog',
        # Set to None to use adaptive linthresh
        "linthresh": 0.2,
        # If using adaptive linthresh, the minimum value it can take
        "_linthreshmin": 1,
        # If using adaptive linthresh, the percentile it should take
        "_linthreshper": 5,
        "linportion": 0.33,
        "ncontours": 100,
        "multiplier": 1.0,
        "zeroinitial": False,
        "plottype": 'phi',
        "repeat": False,
        "xaxis": 'z',
        "yaxis": 'x',
        "slicepos": None,
        "labelsize": 24,
        "legendsize": 12,
        "titlesize": 32,
        "ncontour_lines": 15,
        "cmap": 'inferno',
        "default_ticks": False,
        "draw_cbar": True,
        "draw_surface": False,
        "draw_image": False,
        "draw_contourlines": True,
        "draw_fieldlines": False,
        "animated": False,
        "valmin": None,
        "valmax": None,
        "offset": 0.0,
        "templates": {
            'phi': {
                'titlestr': 'Electrostatic potential',
                'titleunits': 'V',
                'linthresh': 0.2,
                'linportion': 0.33,
            },
            'E': {
                'titlestr': 'Electric field magnitude',
                'titleunits': 'V/nm',
                'linthresh': None,
                '_linthreshmin': 1e-20,
                '_linthreshper': 1,
                'linportion': 0.01,
            },
            'rho': {
                'titlestr': 'Electron density',
                'titleunits': 'cm${}^{-3}$',
                'linthresh': None,
                '_linthreshmin': 1,
                '_linthreshper': 5,
                'linportion': 0.05,
            },
            'barrier': {
                'title': 'Potential energy profiles',
                'ylabel': 'Barrier index (eV)',
                'multiplier': -1.0,
                'zeroinitial': True,
            },
            'image': {
                'draw_image': True,
                'default_ticks': True,
                'draw_contourlines': False,
                'scale': 'linear',
            },
        }
    }

    styles = {
        'arvind': {
            'labelsize': 'medium',
            'legendsize': 'medium',
            'titlesize': 'large',
            'cmap': 'viridis',
            'draw_image': True,
            'draw_contourlines': False,
            'default_ticks': True,

            'templates': {
                'phi': {'titleunits': 'volts', 'scale': 'linear'},
                'E': {'titleunits': 'V nm$^{-1}$', 'linportion': 0.1},
                'rho': {'linportion': 0.1}
            }
        },
        'roelof': {
            'labelsize': 'medium',
            'legendsize': 'medium',
            'titlesize': 'large',
            'cmap': 'viridis',
            'draw_image': True,
            'draw_contourlines': False,
            'default_ticks': True,
            'templates': {
                'barrier': {'zeroinitial': False},
                'phi': {'slicepos': 2.5e-7},
                'E': {'titleunits': 'V nm$^{-1}$', 'linportion': 0.1}
            }
        }
    }

    def __init__(self, siminfo, array, template=None, plot1d=False,
                 ax=None, style=None, **kwargs):
        """Plot given array data.
        Arguments:
            siminfo (runinfo.SimInfo): Object with the simulation info (eg
                simulation size)
            array (np.ndarray): Numpy array containing the data to plot.
            template (string): One of 'phi', 'E', 'rho', 'barrier', or None;
                used to determine default labels. Default None.
            plot1d (bool): If True, plot 1D cuts instead of normal plots. Note
                the xaxis is used in this case as the plotted axis, and the
                yaxis becomes the swept axis.
            ax (matplotlib.Axes): If specified, plot on this axes object. If
                unspecified, get current axes.
            style (string): If not None, override default parameters with a
                pre-defined style set. Currently 'arvind' is implemented.
                Manually supplied parameters will override the defaults in the
                style.
            numpoints (int): If plot1d is True; number of cuts to use
            points (float or list of floats): If plot1d is True, position(s) (in
                m) to plot. If None, plot equally spaced positions. If this is
                used, it overrides numpoints.
            draw_contourlines (bool): If True, draw contour lines. Default True.
            draw_fieldlines (bool): If True, draw field lines. Default False.
            xlabel (string), abscissa label
            ylabel (string), ordinate label
            title (string): If specified, use only this title
            titlestr (string): If title is not specified, the name of the
                quantity to be used in the title.
            titleunits (string): If title is not specified, the name of the
                units to be used for the title.
            titleline2 (string): If title is not specified, use as optional 2nd
                line of title.
            labelsize (int), default 24
            titlesize (int), default 32
            scale (string): Either 'linear' or 'symlog'. The 'linthresh' and
                'linportion' parameters are ignored if using a linear scale.
            linthresh (float), point where linear vs log scaling transition
                occurs. Default 0.2, always positive
            linportion (float), fraction of the scale to allot to the linear
                portion.  Default 0.33, always positive
            ncontours (int), number of contours to use in filled gradations.
                Default 100
            ncontour_lines (int), number of contour lines; also number of tick
                labels on color bar. Default 15.
            multiplier (float), default 1.0, for multiplying the array by a
                factor, e.g. -1.0 to see negative potentials.
            zeroinitial (bool): If True, for 1D cuts only, perform a vertical
                shift such that leftmost point is at y=0.
            cmap (string): String for a matplotlib colormap. Default inferno.
            draw_cbar (bool): If True, draw color bar. Default True.
            draw_surface (bool): If True, use ax.plot_surface() rather than
                ax.contourf. Better display for 3D plots (do not use with 2D).
                Default False.
            draw_image (bool): If True, use ax.imshow() rather than
                ax.contourf. Displays pixelation with no interpolation.
                Default False.
            repeat (bool): If True, draw two periods in ordinate direction.
                Default False.
            xaxis (string): Axis along the abscissa, one of
                ['r', 'x', 'y', 'z']. Default 'z'.
            yaxis (string): Axis along the ordinate, one of
                ['r', 'x', 'y', 'z']. Default 'x'.
            slicepos (float): Position to take slice along 3rd dimension for 3D
                (m). Default 0.
            animated (bool): If True, set up the plot for an animation rather
                than a static image. Defaults to False, and ignored unless
                draw_image is True.
            valmin (float): If not None, override default minimum of the array
                being plotted for color scales etc.
            valmax (float): If not None, override default maximum of the array
                being plotted for color scales etc.
            offset (float): Constant offset added to values, e.g. for changing
                potential zero. Applied after the multiplier.
        """
        self.style = style
        self.template = template
        self.params = copy.copy(self.param_defaults)
        if style is not None:
            self.params = util.recursive_update(
                self.params, self.styles[style])

        self.siminfo = siminfo
        self.array = array
        self.plot1d = plot1d
        if self.template is not None:
            self.params.update(self.params['templates'][self.template])
        self.params.update(**kwargs)

        self.valmin = self.params['valmin']
        self.valmax = self.params['valmax']
        if self.valmin is None:
            self.valmin = np.min(self.array)
        if self.valmax is None:
            self.valmax = np.max(self.array)

        if self.params['linthresh'] is None:
            self.params['linthresh'] = 10**(int(np.log10(max(
                self.params['_linthreshmin'],
                np.percentile(np.unique(np.abs(array)),
                              self.params['_linthreshper'])))))
        # Decide if it should be linear: linportion is large, or linthresh is
        # large, or decades is < 1.
        decades = (
            np.log10(max(self.params["linthresh"], self.valmax))
            + np.log10(max(self.params["linthresh"], -self.valmin))
            - 2*np.log10(self.params["linthresh"]))
        if (self.params['linportion'] >= 1) or (decades < 1.0):
            self.params['scale'] = 'linear'
        if self.params['scale'] == 'linear':
            self.norm = colors.Normalize(vmin=self.valmin,
                                         vmax=self.valmax)
        else:
            # Number of decades for the linear portion to be equivalent to
            lindecades = ((self.params["linportion"] /
                           (1 - self.params["linportion"]))*decades)
            self.norm = colors.SymLogNorm(
                linthresh=np.abs(self.params["linthresh"]), linscale=lindecades,
                vmin=self.valmin, vmax=self.valmax, base=10
            )

        if ax is None:
            ax = plt.gca()
        self.ax = ax

        self.slice_array()
        self.mod_array()
        self.set_plot_labels()

        if self.plot1d:
            self.plot_1d()
        else:
            self.plot_2d()

    def slice_array(self):
        self.dim = len(self.array.shape)
        print('DIM ', self.dim)
        xaxis_idx, yaxis_idx, sliceaxis_idx, self.sliceaxis_str = (
            util.get_axis_idxs(self.params["xaxis"], self.params["yaxis"],
                               self.dim))
        self.xaxisvec = self.siminfo.get_vec(self.params["xaxis"])
        if self.dim == 1:
            z_span = max(self.xaxisvec) - min(self.xaxisvec)
            self.yaxisvec = np.linspace(-z_span/10., z_span/10., 2)
        else:
            self.yaxisvec = self.siminfo.get_vec(self.params["yaxis"])
        self.slicevec = (self.siminfo.get_vec(sliceaxis_idx)
                         if self.dim == 3 else None)
        self.array = util.get_2D_field_slice(
            self.array, xaxis_idx, yaxis_idx, self.slicevec,
            self.params["slicepos"])

    def mod_array(self):
        if self.params["repeat"]:
            self.yaxisvec = np.concatenate(
                [self.yaxisvec,
                 self.yaxisvec + self.yaxisvec[-1] - self.yaxisvec[0] + 1e-9])
            self.array = np.vstack((self.array, self.array))
        self.array = (
            self.array[:, :]*self.params["multiplier"] + self.params["offset"]
        )

    def set_plot_labels(self):
        if self.params["multiplier"] == 1.0:
            if self.params['titleunits'] != '':
                maintitle = "{} ({})".format(self.params["titlestr"],
                                             self.params["titleunits"])
            else:
                maintitle = self.params['titlestr']
        else:
            maintitle = (
                r"{} ({:.1f}$\times${})".format(
                    self.params["titlestr"], self.params["multiplier"],
                    self.params["titleunits"]))

        if self.params["xlabel"] is None:
            self.params["xlabel"] = r"{} ($\mu$m)".format(self.params["xaxis"])
        if self.params["ylabel"] is None:
            # For 1D, full string with units goes on y-axis
            if self.plot1d:
                self.params["ylabel"] = maintitle
                maintitle = self.params["titlestr"]
            else:
                self.params["ylabel"] = r"{} ($\mu$m)".format(
                    self.params["yaxis"])

        if self.params["title"] is None:
            self.params["title"] = maintitle
            if self.params["titleline2"] is not None:
                self.params["title"] += "\n" + self.params["titleline2"]
            if self.dim == 3 and self.params["slicepos"] is not None:
                self.params["title"] += r"\n{} = {:.3g} $\mu$m".format(
                    self.sliceaxis_str, self.params["slicepos"]*1e6)

        self.ax.set_xlabel(self.params["xlabel"],
                           fontsize=self.params["labelsize"])
        self.ax.set_ylabel(self.params["ylabel"],
                           fontsize=self.params["labelsize"])
        self.ax.set_title(self.params["title"],
                          fontsize=self.params["titlesize"])

    def plot_1d(self):
        if self.params["sweepaxlabel"] is None:
            self.params["sweepaxlabel"] = self.params["yaxis"]
        if self.params["points"] is None:
            self.params["points"] = np.linspace(
                np.min(self.yaxisvec), np.max(self.yaxisvec),
                self.params["numpoints"])
        idx_list = []
        barrier_indices = []
        for point in util.return_iterable(self.params["points"]):
            idx_list.append(np.argmin(np.abs(self.yaxisvec - point)))
        for idx in idx_list:
            spos = self.yaxisvec[idx]
            cut = self.array[idx, :]

            # Reference to leftmost point
            if self.params["zeroinitial"]:
                cut = cut - cut[0]

            self.ax.plot(self.xaxisvec*1e6, cut,
                         label=r"{} = {:.3g} $\mu$m".format(
                             self.params["sweepaxlabel"], spos*1e6))
            barrier_indices.append(np.amax(cut))

        if self.template == 'barrier':
            barrier_index = np.amin(barrier_indices)
            print('Anode barrier index = %.3f eV' % barrier_index)
            self.ax.plot(self.xaxisvec * 1e6,
                         barrier_index * np.ones_like(self.xaxisvec), '--k',
                         label='minimum barrier = %.3f eV' % barrier_index)

        self.ax.legend(fontsize=self.params["legendsize"])

    def plot_2d(self):
        # SymLogNorm has a linear portion and a log portion. Set up this norm
        # and choose contours in both regions.
        norm = self.norm
        contour_points = self._gen_plot_contours()
        print("CONTOUR POINTS SHAPE ", contour_points.shape)
        print("xaxisvec", len(self.xaxisvec))
        print("yaxisvec", len(self.yaxisvec))
        [X, Y] = np.meshgrid(self.xaxisvec, self.yaxisvec)
        # Draw filled contours
        print("SHAPE ", self.array[:, :].shape)
        print("X SHAPE", X.shape)
        print("Y SHAPE", Y.shape)
        if self.params["draw_surface"]:
            self.contours = self.ax.plot_surface(X*1e6, Y*1e6, self.array[:, :],
                                                 norm=norm,
                                                 cmap=self.params["cmap"])
        elif self.params["draw_image"]:
            self.contours = self.ax.imshow(self.array[:, :], norm=norm,
                                           cmap=self.params["cmap"],
                                           origin='lower',
                                           extent=(np.min(X)*1e6, np.max(X)*1e6,
                                                   np.min(Y)*1e6,
                                                   np.max(Y)*1e6),
                                           animated=self.params['animated'])
        else:
            self.contours = self.ax.contourf(X*1e6, Y*1e6, self.array[:, :],
                                             contour_points, norm=norm,
                                             cmap=self.params["cmap"])
        self.ax.axis('scaled')
        if self.params["draw_cbar"]:
            cbar = plt.colorbar(self.contours, spacing='proportional',
                                ax=self.ax)
            if not self.params["default_ticks"]:
                cbar.set_ticks(
                    [contour_points[ii] for ii in range(len(contour_points))])
                cbar.set_ticklabels([
                    "{:.2g}".format(contour_points[ii])
                    if ii % (len(contour_points)
                             // self.params["ncontour_lines"]) == 0
                    else "" for ii in range(len(contour_points))])
        if self.params["draw_contourlines"]:
            contours_drawn = [
                contour_points[ii] for ii in range(len(contour_points))
                if ii % (len(contour_points)
                         // self.params["ncontour_lines"]) == 0]
            linec = self.ax.contour(X*1e6, Y*1e6, self.array[:, :],
                                    contours_drawn,
                                    norm=norm, colors='black', linewidths=1)
            self.ax.clabel(linec, colors='w', fmt='%.2g')
        if self.params["draw_fieldlines"]:
            grad = np.gradient(self.array)
            self.ax.streamplot(X*1e6, Y*1e6, grad[1], grad[0], density=2.0,
                               linewidth=1, color="blue", arrowstyle='->',
                               arrowsize=1.5)

    def _gen_plot_contours(self):
        """Generate the list of contours."""
        if self.params['scale'] == 'linear':
            return np.linspace(self.valmin, self.valmax,
                               self.params['ncontours'])
        # Make sure end points are captured, rather than missed due to floating
        # point errors.
        pmin = self.valmin - 1e-5*(self.valmax - self.valmin)
        pthresh = np.abs(self.params["linthresh"])
        pmax = self.valmax + 1e-5*(self.valmax - self.valmin)
        # Try to have a proportional set of negative log contours, linear
        # contours, and positive log contours. Linear is always 'linportion' of
        # total # of contours here.
        # Plot linear contours (any values in linear range)?
        if pmin < pthresh and pmax > -pthresh:
            lincontours = np.linspace(
                max(pmin, -pthresh), min(pmax, pthresh),
                int(round(self.params["ncontours"]*self.params["linportion"])))
        else:
            lincontours = np.array([])
        nlogcontours = int(round(self.params["ncontours"] - len(lincontours)))
        # Plot negative logarithmic contours?
        if pmin < -pthresh:
            nnegcontours = max(0, int(round(
                ((np.log(-pmin) - np.log(pthresh))
                 / (np.log(-pmin) + np.log(max(pthresh, abs(pmax)))
                    - 2.*np.log(pthresh))
                 * nlogcontours))))
            neglogcontours = -1.0*np.exp(
                np.linspace(np.log(pthresh), np.log(abs(pmin)), nnegcontours))
        else:
            nnegcontours = 0
            neglogcontours = np.array([])
        # Plot positive logarithmic contours?
        if pmax > pthresh:
            nposcontours = max(0, nlogcontours - nnegcontours)
            poslogcontours = np.exp(np.linspace(np.log(pthresh), np.log(pmax),
                                                nposcontours))
        else:
            poslogcontours = np.array([])
        # Multiply by the appropriate factor, eliminate duplicates with set,
        # then sort
        contour_points = sorted(set(
            np.concatenate([neglogcontours, lincontours, poslogcontours])))
        return contour_points