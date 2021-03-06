#!/usr/bin/env python
# coding: utf-8

# This file is part of the OMI Python package.
#
# The OMI Python package is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# The OMI Python Package is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the OMI Python Package. If not, see
# <http://www.gnu.org/licenses/>.

from __future__ import division
from __future__ import print_function

import json
import os
import warnings

import numpy as np
import scipy.interpolate

import omi.he5
import omi.pixel
import omi.psm

#import pyximport; pyximport.install()
import omi.cgrate

PACKAGE_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

verbosity = 0

class Grid(object):
    def __init__(self, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon,
        resolution):
        """\
        A longitude-latitude grid which is used to store values, errors
        and weights of the Level 3 product.

        Parameter
        ---------
        llcrnrlat
            latitude of lower-left corner

        urcrnrlat
            latitude of upper-right

        llcrnrlon
            longitude of lower-left corner

        urcrnrlon
            longitude of upper-right corner

        resoluton
            resolution of grid
        """
        self.llcrnrlat = llcrnrlat
        self.urcrnrlat = urcrnrlat
        self.llcrnrlon = llcrnrlon
        self.urcrnrlon = urcrnrlon
        self.resolution = resolution

        self.lon = np.arange(llcrnrlon, urcrnrlon, self.resolution, dtype='float64')
        self.lat = np.arange(llcrnrlat, urcrnrlat, self.resolution, dtype='float64')

        self._shape = (self.lon.size, self.lat.size)

        self.values = np.zeros(self.shape, dtype='float64')
        self.errors = np.zeros(self.shape, dtype='float64')
        self.weights = np.zeros(self.shape, dtype='float64')


    @classmethod
    def by_name(cls, gridname, filename=None):
        """\
        Create omi.Grid by `gridname` as defined in a JSON file
        `filename`.
        """
        if filename is None:
            filename = os.path.join(PACKAGE_DATA_FOLDER, 'grids.json')

        with open(filename) as f:
            var = json.load(f)[gridname]

        return cls(*var)


    @property
    def shape(self):
        """
        Return the dimensions of the grid (lon x lat)
        :return: tuple, (lon.size, lat.size)
        """
        # The previous version that calculated the shape from the corner points was subject to rounding error if the
        # end point isn't the next grid point. (e.g. lon = [-124.975, -65.0], lat = [25.025, 50]. That caused an error
        # in cgrate when the lon vector (in this case) was longer than the shape in the longitudinal direction, since
        # the values had the wrong shape.
        return self._shape


    def save_as_he5(self, filename):
        """\
        Save Grid to `filename` as HDF5 file.
        """
        data = [
            ('lon', self.lon),
            ('lat', self.lat),
            ('values', self.values),
            ('errors', self.errors),
            ('weights', self.weights)
        ]
        omi.he5.write_datasets(filename, data)


    def save_as_image(self, filename, vmin=None, vmax=None):
        """\
        Save Grid to `filename` as image
        (requires matplotlib and basemap)
        """
        try:
            import matplotlib as mpl
            mpl.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.basemap import Basemap
        except ImportError as e:
            print("No matplotlib and/or basemap to create 'png' image.")
            raise e

        fig = plt.figure(figsize=(8, 8*self.ratio))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

        m = Basemap(ax=ax, resolution='i', **self.to_basemap())
        m.drawcoastlines() 
        m.drawcountries()   # Zusatz von Annette
        #m.drawrivers()     # Zusatz von Annette
        
        
        res = m.imshow(self.values.T, vmin=vmin, vmax=vmax)

        fig.savefig(filename, dpi=500)
        plt.close(fig)


    def norm(self):
        """\
        Normalise values and errors.
        """
        no_values = (self.weights == 0.0)
        self.weights[no_values] = np.nan

        self.values /= self.weights
        self.errors = np.sqrt(self.errors) / self.weights



    def zero(self):
        """\
        Set values, errors and weights to zero.
        """
        self.values[...] = 0.0
        self.errors[...] = 0.0
        self.weights[...] = 0.0


    @property
    def ratio(self):
        return (self.urcrnrlat - self.llcrnrlat) / (self.urcrnrlon - self.llcrnrlon)


    def to_basemap(self):
        """\
        Return corners of grid, which can be used as parameter for
        mpl_toolkits.Basemap:
        >>> from mpl_toolkits.basemap import Basemap
        >>> grid = Grid.by_name('asia')
        >>> m = Basemap(**grid.to_basemap())
        """
        return {
            'llcrnrlat': self.llcrnrlat,
            'urcrnrlat': self.urcrnrlat,
            'llcrnrlon': self.llcrnrlon,
            'urcrnrlon': self.urcrnrlon
        }



def interp_missing_values(values, dx, dy):
    """\
    Interpolate missing values using linear interpolation for points within
    the convex hull of available data. Data points outside are interpolated
    by the nearest neighbor method.

    Parameter
    ---------
    values : np.ma.array, shape(M,N)
        measurement values (missing values are masked)

    dx, dy : array_like, shape(M,N)
        pixel sizes in axross-track (dx) and along-track (dy)
        direction

    """
    mask = values.mask
    values = values.data

    x = dx.cumsum(1)
    y = dy.cumsum(0)

    # linear interpolation
    missing_points = np.concatenate(([x[mask]],[y[mask]]), axis=0).T
    known_points = np.concatenate(([x[~mask]],[y[~mask]]), axis=0).T

    values[mask] = scipy.interpolate.griddata(
        known_points, values[~mask], missing_points,
        method='linear', fill_value=np.nan
    )
    values = np.ma.array(values, mask=np.isnan(values))

    # nearest neighbour interpolation
    if np.any(values.mask):

        mask = values.mask
        values = values.data

        missing_points = np.concatenate(([x[mask]],[y[mask]]), axis=0).T
        known_points = np.concatenate(([x[~mask]],[y[~mask]]), axis=0).T

        values[mask] = scipy.interpolate.griddata(
            known_points, values[~mask], missing_points,
            method='nearest'
        )
        values = np.ma.array(values, mask=np.isnan(values))

    return values



def geo_to_grid(grid, lon, lat):
    """\
    Convert geographic coordinates (lon/lat) to
    grid indices.
    """

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    lon = np.array((lon - grid.llcrnrlon) / grid.resolution, dtype='float64')
    lat = np.array((lat - grid.llcrnrlat) / grid.resolution, dtype='float64')

    return lon, lat



def psm_grid(grid, lon_center, lat_center, lon_corner, lat_corner,
    values, errors, stddev, weights, missing_values,
    lon_spacecraft, lat_spacecraft, alt_spacecraft,
    gamma, rho_est=1e16, lut=None):
    """\
    Grid values using PSM.

    Parameter
    ---------
    grid : omi.Grid
        a longitude-latitude grid for storing
        gridded values, errors and weights

    lon_center : array_like, shape(M,N)
        longitude of pixel center

    lat_center : array_like, shape(M,N)
        latitude of pixel center

    lon_corner : array_like, shape(4,M,N)
        longitudes of tiled pixel corners

    lat_corner : array_like, shape(4,M,N)
        latitudes of tiled pixel corners

    values : array_like, shape(M,N)
        measurement values

    errors : array_like, shape(M,N)
        measurement errors

    stddev : array_like, shape(M,N)
        estimated standard deviation of measurement
        values

    missing_values : array_like, bool, shape(M,N)
        mask for missing values (= True)

    lon_spacecraft : array_like, shape(M,)
        longitude of spacecraft

    lat_spacecraft : array_like, shape(M,)
        latitude of spacecraft

    alt_spacecraft : array_like, shape(M,)
        altitude of spacecraft (in km)

    gamma : array_like, shape(N,) or shape(1,)
        smoothing parameter

    rho_est : float
        estimate of typical maximum value
        of distribution (the value is also
        used to scale `values` and `stddev`)

    lut : omi.psm.MMatrixLUT or None, 'none'
        look-up table for entries of measurement
        matrix M as function of distance between
        spacecraft and ground pixel.

        If `lut is None` (default), a pre-calculated
        LUT is loaded from the package folder.

        If `lut == "none"` (a string!), the M matrix
        entries will be computed by numerical
        integration. That is very(!) slow!


    Returns
    -------
    The same `grid` object as passed as parameter.

    """

    # distance and exposure time
    exposure_time = 2.0
    distances = omi.pixel.compute_distance(
        lon_center, lat_center,
        lon_spacecraft, lat_spacecraft, alt_spacecraft
    )

    # gamma and LUT for matrix M (kappa)
    if lut is None:
        lut = omi.psm.MMatrixLUT(PACKAGE_DATA_FOLDER)
    elif lut == 'none':
        lut  = None

    # compute grid distances
    dx, dy = omi.pixel.compute_pixel_size(lon_corner, lat_corner, only_dx=True)

    # interpolate missing data
    missing_values = values.mask.copy()
    values = interp_missing_values(values, dx, dy)
    stddev = stddev.data.copy()
    stddev[missing_values] = rho_est

    # compute coeffiencts of PSM spline
    p, d, qx, qy, alpha, beta = omi.psm.parabolic_spline_algorithm(values, stddev, dx, dy,
        gamma, rho_est, distances, exposure_time, missing_values, lut=lut)

    # grid using PSM spline
    lon_corner_grid, lat_corner_grid = geo_to_grid(grid, lon_corner, lat_corner)

    grid.values = omi.cgrate.draw_orbit(
        grid.lon, grid.lat, grid.values, grid.errors, grid.weights,
        lon_corner_grid, lat_corner_grid,
        lon_corner, lat_corner,
        values, errors, weights, np.array(missing_values, int), alpha, beta,
        p, d, qx, qy,
        'psm'
    )

    return grid


def cvm_grid(grid, lon_corner, lat_corner, values, errors,
    weights, missing_values, is_flag=False):
    """\
    Grid values and errors using CVM.

    Parameter
    ---------
    grid : omi.Grid
        a longitude-latitude grid for storing
        gridded values, errors and weights

    lon_corner : array_like, shape(4,M,N)
        longitudes of pixel corners

    lat_corner : array_like, shape(4,M,N)
        latitudes of pixel corners

    values : array_like, shape(M,N)
        measurement values

    errors : array_like, shape(M,N)
        measurement errors

    missing_values : array_like, bool, shape(M,N)
        mask for missing values (= True)


    Returns
    -------
    The same `grid` object as passed as parameter.

    """
    lon_corner_grid, lat_corner_grid = geo_to_grid(grid, lon_corner, lat_corner)

    if is_flag:
        method_name = 'cvm-flags'
    else:
        method_name = 'cvm'

    grid.values = omi.cgrate.draw_orbit(
        grid.lon, grid.lat, grid.values, grid.errors, grid.weights,
        lon_corner_grid, lat_corner_grid,
        None, None,
        values, errors, weights, np.array(missing_values, int),
        None, None,
        None, None, None, None,
        method_name
    )

    return grid


def compute_smoothing_parameter(gamma_at_nadir, gamma_at_edge):
    """\
    Compute smoothing parameter (gamma) as function
    of pixel size (overlap).

    Parameter
    ---------
    `gamma_at_nadir` and `gamma_at_edge`.

    Returns
    -------
    gamma, np.array, shape(N=60,)
    """

    # precomputed overlaps for across-track position 0..59
    overlap = np.array([
            0.59614826,  0.48260896,  0.40723757,  0.34840238,  0.30109656,
            0.2616919 ,  0.22800451,  0.19887041,  0.17358065,  0.15160903,
            0.13258199,  0.11614961,  0.10198027,  0.08979957,  0.07929943,
            0.07029387,  0.06258653,  0.05594367,  0.0502595 ,  0.04537765,
            0.04115662,  0.0375653 ,  0.03452365,  0.03194877,  0.0297819 ,
            0.02798369,  0.02653643,  0.02542291,  0.02462016,  0.02409373,
            0.02389364,  0.02400853,  0.02441251,  0.02514975,  0.02625402,
            0.02774706,  0.02965554,  0.03202502,  0.03491346,  0.03837971,
            0.04251103,  0.04735523,  0.05296155,  0.05947557,  0.06700813,
            0.07567288,  0.08563873,  0.09707048,  0.11012513,  0.12506143,
            0.14220217,  0.16190492,  0.18467912,  0.2112874 ,  0.24278989,
            0.2806403 ,  0.32668071,  0.38283713,  0.45154673,  0.55376118
    ])

    # linear interpolation
    m = (gamma_at_edge - gamma_at_nadir) / (overlap[0] - overlap[30])
    b = gamma_at_nadir - m * overlap[30]

    return m * overlap + b






def clip_orbit(grid, lon, lat, data, boundary=(2,2)):
    """\
    Clip orbit to grid domain.

    Parameter
    ---------
    grid : omi.Grid
        a longitude-latitude grid for storing
        gridded values, errors and weights

    lon/lat : array_like, shape(M,N) or shape(4,M,N)
        longitude and latitude of pixel centers or corners

    data : dict(key->np.ndarray)
        OMI data e.g. loaded with `omi.he5.read_datasets`

    boundary : tuple (default: (2,2))
        Number of extra pixels (x-direction and y-direction)
        which are not clipped, if they are outside of the grid
        domain.

    Returns
    -------
    The clipped datafields as dictionary (key->dataset). An
    additional np.array ('ColumnIndices') is added, which gives
    the indices of the rows in across-track direction.

    """
    domain = mask_grid_domain(grid, lon, lat)

    if lon.ndim == 3:
        across_size = lon.shape[2]
        along_size = lon.shape[1]
    elif lon.ndim == 2:
        across_size = lon.shape[1]
        along_size = lon.shape[0]
    else:
        raise RuntimeError('lon is expected to be 2D or 3D, it is not')

    if across_size != 60 and across_size != 30:
        warnings.warn('Across track size is {0}, this is unusual (typically is 60 or 30)'.format(across_size))

    if np.any(domain):
        data, domain, col_indices = remove_out_of_domain_data(data, domain, boundary, across_size, along_size)
        data['ColumnIndices'] = col_indices

    else:
        for key in data:
            data[key] = np.array([])
        data['ColumnIndices'] = np.array([])

    return data




def mask_grid_domain(grid, lon, lat, type_='any'):
    """\
    Mask area where lon/lat is within grid boundaries.
    """
    lon, lat = geo_to_grid(grid, lon, lat)

    if lon.ndim == 3 and lat.ndim == 3:
        if type_ == 'any':
            mask = np.any(
                  (0 <= lon) & (lon <= grid.shape[0])
                & (0 <= lat) & (lat <= grid.shape[1]),
            axis=0)

        elif type_ == 'all':
            mask = np.all(
                  (0 <= lon) & (lon <= grid.shape[0])
                & (0 <= lat) & (lat <= grid.shape[1]),
            axis=0)

    else:
        mask = (
              (0 <= lon) & (lon <= grid.shape[0])
            & (0 <= lat) & (lat <= grid.shape[1])
        )

    return mask



def remove_out_of_domain_data(data, domain, boundary, across_size, along_size):
    """\
    Remove data from datasets which are outside domain
    with an added boundary in x- and y-direction.

    See `omi.clip_orbit` for details.
    """

    # get min/max i:
    indices = domain.any(1).nonzero()[0]
    y_slice = slice(
        max(min(indices-boundary[1]), 0),
        max(indices + boundary[1]) + 1
    )

    indices = domain.any(0).nonzero()[0]
    x_slice = slice(
        max(min(indices-boundary[0]), 0),
        min(max(indices+boundary[0]) + 1, 60)
    )

    col_indices = np.arange(x_slice.start, x_slice.stop)
    domain = domain[y_slice,x_slice]

    for name, field in data.items():

        if field.ndim == 1:
            if verbosity > 1:
                print('  omi.remove_out_of_domain_data: {0} is a 1D variable'.format(name))

            if field.size == across_size:
                data[name] = field[x_slice]
            else:
                data[name] = field[y_slice]

        elif field.ndim == 2:
            if verbosity > 1:
                print('  omi.remove_out_of_domain_data: {0} is a 2D variable'.format(name))

            if (field.size == across_size or field.size == along_size):
                if along_size > 1:
                    raise RuntimeError('{0} size equals across or along track dimensions, this suggests it is actually a '
                                       '1D variable that has a singleton dimensions.'.format(name))
                else:
                    print('    omi.remove_out_of_domain_data: Size of 2D variable {0} equals along or across track dimensions. Since the along track'
                                  ' dimension is 1, this is likely okay, unless {0} is supposed to be a 1D variable'.format(name))

            data[name] = field[y_slice, x_slice]

        elif field.ndim == 3:
            if verbosity > 1:
                print('  omi.remove_out_of_domain_data: {0} is a 3D variable'.format(name))

            # find dim for across-track direction
            if across_size == along_size:
                # If the along size is the same as the across size, we have to be more careful because index() always
                # returns the first instance of a value in a list. Since we assume that across track always follows
                # along track, find the second occurance of that size.
                tmp_inds = [i for i in range(len(field.shape)) if field.shape[i] == across_size]
                if len(tmp_inds) < 2:
                    raise IndexError('3d field "{0}" was expected to have the along and across track lengths the same, but I only found one dimension of that length'.format(name))
                else:
                    across = tmp_inds[1]
            else:
                try:
                    across = field.shape.index(across_size)
                except ValueError:
                    raise IndexError('3d field "{0}" has no x-track axis ({1} columns)'.format(name, across_size))


            # We assume that it is ordered [along, across, other] or [other, along, across], check that assumption
            if field.shape[across - 1] != along_size:
                raise IndexError('3d field "{0}" does not have the across-track dimension follow the along track one'
                                 .format(name))

            if across == 1:
                data[name] = field[y_slice, x_slice,:]

            elif across == 2:
                data[name] = field[:,y_slice, x_slice]

            else:
                raise NotImplementedError('Do not know how to clip a variable with the across dimension == {}'.format(across))

        else:
            raise NotImplementedError('Do not know how to clip a {0} dimension variable ({1})'.format(field.ndim, name))

    return data, domain, col_indices



if __name__ == '__main__':
    pass

