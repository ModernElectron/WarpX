import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy
import collections

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
            self.params = recursive_update(
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

        if self.valmin >= self.valmax:
            self.valmax = self.valmin + 1e-10

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
            get_axis_idxs(self.params["xaxis"], self.params["yaxis"],
                               self.dim))
        self.xaxisvec = self.siminfo.get_vec(self.params["xaxis"])
        if self.dim == 1:
            z_span = max(self.xaxisvec) - min(self.xaxisvec)
            self.yaxisvec = np.linspace(-z_span/10., z_span/10., 2)
        else:
            self.yaxisvec = self.siminfo.get_vec(self.params["yaxis"])
        self.slicevec = (self.siminfo.get_vec(sliceaxis_idx)
                         if self.dim == 3 else None)
        self.array = get_2D_field_slice(
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
        for point in return_iterable(self.params["points"]):
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
        print("contour points", contour_points)
        #print("CONTOUR POINTS SHAPE ", contour_points.shape)
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

        plt.show()
        plt.savefig("fig.png")

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

axis_labels_2d = ['r', 'x', 'z']
axis_labels_3d = ['x', 'y', 'z']
axis_dict_3d = {'x': 0, 'y': 1, 'z': 2}
axis_dict_2d = {'r': 0, 'x': 0, 'z': 1}


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
            if all([(isinstance(y, collections.abc.Iterable) and not isinstance(y, str))
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
