from __future__ import print_function

import datetime
from glob import glob
import numpy as np
import numpy.ma as ma
import os
import pdb
import re
from scipy.spatial.qhull import QhullError

from ospy.hdf5 import saveh5 # debugging only

# This must be built and installed from the omi subdirectory. See omi/help.txt for instructions.
import omi

__author__ = 'Josh'
__verbosity__ = 1
__save_swath__ = False  # debugging only, remove the clauses controlled by that when you're done

# Define the datasets that need to be loaded from BEHR files
behr_datasets = ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
                 'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude',
                 'Latitude', 'Longitude', 'SpacecraftAltitude', 'SpacecraftLatitude', 'SpacecraftLongitude',
                 'CloudRadianceFraction', 'CloudPressure', 'BEHRColumnAmountNO2Trop', 'ColumnAmountNO2Trop',
                 'ColumnAmountNO2TropStd', 'BEHRAMFTrop', 'VcdQualityFlags', 'XTrackQualityFlags', 'SolarZenithAngle', 'Time']


def behr_version():
    """
    Grab the BEHR version string from the MATLAB function BEHR_version.m. Assumes that this file is located in a
    subdirectory of the BEHR repository
    :return: the version string
    """
    mydir = os.path.realpath(os.path.dirname(__file__))
    #vers_file = os.path.join(mydir, '..', 'Utils', 'Constants', 'BEHR_version.m')

    # temporary, for testing
    vers_file = os.path.join(mydir, '..', 'BEHR', 'Utils', 'Constants', 'BEHR_version.m')
    with open(vers_file, 'r') as f:
        first_line = True # need to skip the first line since it's just the function definition
        for line in f:
            if first_line:
                first_line = False
                continue

            # Remove anything after a % (matlab comment)
            comment = line.find('%')
            line = line[:comment] if comment >= 0 else line

            # look for "ver_str", the variable that is assigned the version string
            if 'ver_str' in line:
                m = re.search('(?<=\').*(?=\')', line)
                if m is None:
                    raise ValueError('Could not find the version string (encompassed by single quotes) in the line containing "ver_str"')
                else:
                    return m.group()


def behr_preprocessing(gridding_method, BEHRColumnAmountNO2Trop, ColumnAmountNO2TropStd, BEHRAMFTrop, SolarZenithAngle, VcdQualityFlags,
                       XTrackQualityFlags, CloudRadianceFraction, FoV75Area, **kwargs):
    """
    Preprocesses a dictionary of datasets from BEHR files. If the dictionary is "data", then it should be passed to this
    function using the ** syntax, i.e. preprocessing(gridding_method, **data). The first input must be the gridding
    method string, which should be 'psm' or 'cvm'.
    :param gridding_method

    The dictionary keys of data will be used as as keywords and the dict values as the values associated with those
    keywords. This helps keep the code here cleaner and makes clear what keys in the dictionary are required, which are:
    :param BEHRColumnAmountNO2Trop
    :param ColumnAmountNO2TropStd
    :param SolarZenithAngle
    :param vcdQualityFlags
    :param XTrackQualityFlags
    :param CloudRadianceFraction
    :param FoV75Area

    kwargs then just holds the remainder and ensures nothing weird happens because we tried to pass arguments with
    keywords that this function doesn't understand.
    :param kwargs

    Returns (from preprocessing):
    :return:    values - masked array of BEHRColumnAmountNO2Trop values
                errors - masked array of ColumnAmountNO2TropStd values
                stddev - masked array of assumed standard deviations that increase linearly with Cloud Rad. Fraction
                weights - masked array of weights as the inverse of the pixel area (FoV75Area) if using PSM, or the
                    inverse of (pixel area * std dev.**2) for CVM.
    """

    # mask bad values - BEHR does not have its own std. deviation value. Very small AMFs typically indicate something
    # went wrong in the processing. This should probably be corrected in the BEHR algorithm itself
    amfmask = (BEHRAMFTrop < 1e-5).filled(True)  # this ensures that masked values remain masked
    cldmask = CloudRadianceFraction > 2.5
    mask = BEHRColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask | amfmask | cldmask

    # mask low quality data
    mask |= SolarZenithAngle > 85
    mask |= VcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags != 0

    return preprocessing(gridding_method=gridding_method, no2_column=BEHRColumnAmountNO2Trop, no2_column_std=ColumnAmountNO2TropStd,
                         cloud_radiance_fraction=CloudRadianceFraction, fov75_area=FoV75Area, mask=mask)

def sp_preprocessing(gridding_method, ColumnAmountNO2Trop, ColumnAmountNO2TropStd, FoV75Area, CloudRadianceFraction,
    SolarZenithAngle, VcdQualityFlags, XTrackQualityFlags, **kwargs):
    """
    Preprocesses a dictionary of datasets from BEHR files, using the NASA standard column as the values instead of the
    BEHR columns. If the dictionary is "data", then it should be passed to this function using the ** syntax, i.e.
    preprocessing(gridding_method, **data). The first input must be the gridding method string, which should be 'psm' or
    'cvm'.
    :param gridding_method

    The dictionary keys of data will be used as as keywords and the dict values as the values associated with those
    keywords. This helps keep the code here cleaner and makes clear what keys in the dictionary are required, which are:
    :param BEHRColumnAmountNO2Trop
    :param ColumnAmountNO2TropStd
    :param SolarZenithAngle
    :param vcdQualityFlags
    :param XTrackQualityFlags
    :param CloudRadianceFraction
    :param FoV75Area

    kwargs then just holds the remainder and ensures nothing weird happens because we tried to pass arguments with
    keywords that this function doesn't understand.
    :param kwargs

    Returns (from preprocessing):
    :return:    values - masked array of BEHRColumnAmountNO2Trop values
                errors - masked array of ColumnAmountNO2TropStd values
                stddev - masked array of assumed standard deviations that increase linearly with Cloud Rad. Fraction
                weights - masked array of weights as the inverse of the pixel area (FoV75Area) if using PSM, or the
                    inverse of (pixel area * std dev.**2) for CVM.
    """
    # mask of bad values
    mask = ColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask

    # mask low quality data
    mask |= SolarZenithAngle > 85
    mask |= VcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags != 0

    return preprocessing(gridding_method=gridding_method, no2_column=ColumnAmountNO2Trop, no2_column_std=ColumnAmountNO2TropStd,
                         cloud_radiance_fraction=CloudRadianceFraction, fov75_area=FoV75Area, mask=mask)


def preprocessing(gridding_method, no2_column, no2_column_std, cloud_radiance_fraction, fov75_area, mask):
    """
    Subordinate preprocessing function that should be called from the specialized preprocessing function for individual
    columns. The specialized preprocessing function should create the mask to remove unwanted values (i.e. row anomaly
    or pixels with processing errors) and then pass the specified values to this function.

    :param gridding_method: the gridding method ('psm' or 'cvm') as a string
    :param no2_column: the tropospheric column to grid as a numpy array
    :param no2_column_std: the standard error of the tropospheric column as a numpy array
    :param cloud_radiance_fraction: the cloud radiance fraction as a numpy array
    :param fov75_area: the FoV75 area as a numpy array; this should have a length of 1 in the along-track dimension and
        length 60 in the across track dimension (except maybe on zoom mode days)
    :param mask: a boolean array that is true where column values should not be used (e.g. row anomaly)

    Returns:
    :return:    values - masked array of BEHRColumnAmountNO2Trop values
                errors - masked array of ColumnAmountNO2TropStd values
                stddev - masked array of assumed standard deviations that increase linearly with Cloud Rad. Fraction
                weights - masked array of weights as the inverse of the pixel area (FoV75Area) if using PSM, or the
                    inverse of (pixel area * std dev.**2) for CVM.
    """

    # set invalid cloud cover to 100% -> smallest weight
    cloud_radiance_fraction[cloud_radiance_fraction.mask] = 1.0

    # VCD values and errors
    values = ma.array(no2_column, mask=mask)
    errors = ma.array(no2_column_std, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al. 2008)
    stddev = 1.5e15 * (1.0 + 3.0 * ma.array(cloud_radiance_fraction, mask=mask))
    area = fov75_area.reshape((1, fov75_area.size))
    area = area.repeat(no2_column.shape[0], axis=0)

    if gridding_method.lower() == 'psm':
        weights = ma.array(1.0 / area, mask=mask)
    elif gridding_method.lower() == 'cvm':
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)
    else:
        raise NotImplementedError('No weighting formula specified for gridding method {}'.format(gridding_method))

    return values, errors, stddev, weights


def generate_filename(save_path, gridding_method, column_product, date_in, date_end=None):
    if date_end is None:
        fname = 'OMI_{3}_{2}_{0}_{1}.he5'.format(behr_version(),
                                                 date_in.strftime('%Y%m%d'),
                                                 gridding_method.upper(),
                                                 column_product.upper())
    else:
        fname = 'OMI_{3}_{4}_{0}_{1}-{2}.he5'.format(behr_version(),
                                                      date_in.strftime('%Y%m%d'),
                                                      date_end.strftime('%Y%m%d'),
                                                      column_product.upper(),
                                                      gridding_method.upper())
    return os.path.join(save_path, fname)


def make_grid(grid_info):
    if isinstance(grid_info, str):
        return omi.Grid.by_name(grid_info)
    elif isinstance(grid_info, dict):
        req_keys = ['llcrnrlat', 'urcrnrlat', 'llcrnrlon', 'urcrnrlon', 'resolution']
        test_bool = [x in grid_info.keys() for x in req_keys]
        if not all(test_bool):
            raise KeyError('If given as a dict, grid_info must have the keys: {0}'.format(', '.join(req_keys)))

        return omi.Grid(**grid_info)
    else:
        raise NotImplementedError('Cannot make a grid from input of type {}'.format(type(grid_info)))



def grid_day_from_file(behr_file, grid_info, grid_method, column_product, verbosity=0):
    day_grid = make_grid(grid_info)
    day_grid.zero() # ensure that values and weights are zeroed (just in case, Annette from Mark's group seems to be
                    # concerned that grids be zeroed out, though it seems like they really should start zeroed)

    for _, orbit_num, data in omi.he5.iter_behr_orbits_in_file(behr_file, behr_datasets):
        if verbosity > 0:
            print(' Gridding orbit no. {0}'.format(orbit_num))
        vals, weights = grid_orbit(data, grid_info, gridding_method=grid_method, column_product=column_product,
                                   verbosity=verbosity)

        if vals is not None:
            day_grid.values += np.nan_to_num(vals) * np.nan_to_num(weights)
            day_grid.weights += weights

    day_grid.norm()
    return day_grid


def grid_day_from_interface(behr_data, behr_grid, grid_method, column_product, verbosity=0):
    if isinstance(behr_data, dict):
        behr_data = [behr_data]
    elif isinstance(behr_data, (list, tuple)):
        if any([not isinstance(x, dict) for x in behr_data]):
            raise TypeError('If given as a list/tuple, all child elements of behr_data must be dicts')
    else:
        raise TypeError('behr_data must be a dict or a list/tuple of dicts')

    if not isinstance(behr_grid, dict):
        raise TypeError('behr_grid must be a dict')

    if not isinstance(grid_method, str):
        raise TypeError('grid_method must be a string')

    if not isinstance(column_product, str):
        raise TypeError('column_product must be a string')

    day_grid = make_grid(behr_grid)
    day_grid.zero()

    for data in behr_data:
        if verbosity > 0:
            print('Gridding swath {} of {}'.format(behr_data.index(data)+1, len(behr_data)))

        if __save_swath__:
            lastfile = sorted(glob('idata-pre_grid-*.he5'))
            if len(lastfile) > 0:
                lastfile = lastfile[-1]
                i_swath = int(re.search('\d\d', lastfile).group())
            else:
                i_swath = 0
            savename = 'idata-pre_grid-{:02d}.he5'.format(i_swath+1)
            print('Saving as', savename)
            saveh5(savename, data=data)

        vals, weights = grid_orbit(data, behr_grid, gridding_method=grid_method, column_product=column_product,
                                   verbosity=verbosity)

        if __save_swath__:
            lastfile = sorted(glob('idata-post_grid-*.he5'))
            if len(lastfile) > 0:
                lastfile = lastfile[-1]
                i_swath = int(re.search('\d\d', lastfile).group())
            else:
                i_swath = 0
            savename = 'idata-post_grid-{:02d}.he5'.format(i_swath+1)
            print('Saving as', savename)
            saveh5(savename, data=data, vals=vals, weights=weights)

        if vals is not None:
            day_grid.values += np.nan_to_num(vals) * np.nan_to_num(weights)
            day_grid.weights += weights

    day_grid.norm()
    return day_grid


def save_individual_days(start_date, end_date, data_path, save_path, grid_info, grid_method, column_product='behr',
                         verbosity=0):
    for filename in omi.he5.iter_behr_filenames(start_date, end_date, data_path):
        if verbosity > 0:
            print('Working on file {0}'.format(filename))
        day_grid = grid_day_from_file(filename, grid_info, grid_method=grid_method, column_product=column_product,
                                      verbosity=verbosity)
        mobj = re.search('\d\d\d\d\d\d\d\d', filename)
        behr_date = datetime.datetime.strptime(mobj.group(), '%Y%m%d')
        save_name = generate_filename(save_path, grid_method, column_product, behr_date)
        day_grid.save_as_he5(save_name)


def multi_day_average(start_date, end_date, data_path, grid_info, grid_method, column_product, verbosity=0):
    avg_grid = make_grid(grid_info)
    avg_grid.zero()
    for filename in omi.he5.iter_behr_filenames(start_date, end_date, data_path):
        if verbosity > 0:
            print('Working on file {0}'.format(filename))

        for _, orbit_num, data in omi.he5.iter_behr_orbits_in_file(filename, behr_datasets):
            if verbosity > 0:
                print(' Gridding orbit no. {0}'.format(orbit_num))

            vals, weights = grid_orbit(data, grid_info, gridding_method=grid_method, column_product=column_product,
                                       verbosity=verbosity)

            if __save_swath__:
                saveh5('data-{:06d}.he5'.format(orbit_num), data=data)
                saveh5('o-{:06d}.he5'.format(orbit_num), vals=vals, weights=weights)

            if vals is not None:
                avg_grid.values += np.nan_to_num(vals) * np.nan_to_num(weights)
                avg_grid.weights += weights

    avg_grid.norm()
    return avg_grid


def save_average(start_date, end_date, data_path, save_path, grid_info, grid_method, column_product='behr', verbosity=0):
    avg = multi_day_average(start_date, end_date, data_path, grid_info, grid_method, column_product=column_product,
                            verbosity=verbosity)
    save_name = generate_filename(save_path, grid_method, column_product, start_date, end_date)
    avg.save_as_he5(save_name)


def grid_orbit(data, grid_info, gridding_method='psm', column_product='behr', verbosity=0):
    # Input checking
    if not isinstance(data, dict):
        raise TypeError('data must be a dict')
    else:
        missing_keys = [k for k in behr_datasets if k not in data.keys()]
        if len(missing_keys) > 0:
            raise KeyError('data is missing the following expected keys: {0}'.format(', '.join(missing_keys)))

    # Ensure 1D datasets are actually 1D by removing any singleton dimensions that Matlab adds because it treats any
    # array as at least 2D.
    for k, v in data.iteritems():
        data[k] = v.squeeze()

    grid = make_grid(grid_info)
    wgrid = make_grid(grid_info)

    # 5) Check for missing corner coordinates, i.e. the zoom product,
    #    which is currently not supported
    if data['TiledCornerLongitude'].mask.any() or data['TiledCornerLatitude'].mask.any():
        return None, None

    # 6) Clip orbit to grid domain
    lon = data['FoV75CornerLongitude']
    lat = data['FoV75CornerLatitude']
    data = omi.clip_orbit(grid, lon, lat, data, boundary=(2, 2))

    if data['BEHRColumnAmountNO2Trop'].size == 0:
        return None, None

    # Preprocess the BEHR data to create the following arrays MxN:
    #    - measurement values
    #    - measurement errors (used in CVM, not PSM)
    #    - estimate of stddev (used in PSM)
    #    - weight of each measurement
    if verbosity > 1:
        print('    Doing {} preprocessing'.format(column_product))
    if column_product.lower() == 'behr':
        values, errors, stddev, weights = behr_preprocessing(gridding_method, **data)
    elif column_product.lower() == 'sp':
        values, errors, stddev, weights = sp_preprocessing(gridding_method, **data)
    else:
        raise NotImplementedError('No preprocessing option for column_product={}'.format(column_product))

    missing_values = values.mask.copy()

    if np.all(values.mask):
        return None, None  # two outputs expected, causes a "None object not iterable" error if only one given

    new_weight = weights / np.sqrt(
        (np.abs((errors / 1e15) * (1 + 2 * data['CloudRadianceFraction'] ** 2))))  # **(1.0/2.0)

    if verbosity > 1:
        print('    Gridding VCDs')

    if gridding_method == 'psm':

        rho_est = 4e16
        gamma = omi.compute_smoothing_parameter(1.0, 10.0)
        try:
            grid = omi.psm_grid(grid,
                                data['Longitude'], data['Latitude'],
                                data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                                values, errors, stddev, weights, missing_values,
                                data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                                data['SpacecraftAltitude'],
                                gamma[data['ColumnIndices']],
                                rho_est
                                )
        except QhullError as err:
            print("Cannot interpolate, QhullError: {0}".format(err.args[0]))
            return None, None

        gamma = omi.compute_smoothing_parameter(40.0, 40.0)
        rho_est = 4

        if verbosity > 1:
            print('    Gridding weights')

        try:
            wgrid = omi.psm_grid(wgrid,
                                 data['Longitude'], data['Latitude'],
                                 data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                                 new_weight, errors, new_weight * 0.9, weights, missing_values,
                                 data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                                 data['SpacecraftAltitude'],
                                 gamma[data['ColumnIndices']],
                                 rho_est
                                 )
            # The 90% of new_weight = std. dev. is a best guess comparing uncertainty
            # over land and sea
        except QhullError as err:
            print("Cannot interpolate, QhullError: {0}".format(err.args[0]))
            return None, None

        grid.norm()  # divides by the weights (at this point, the values in the grid are multiplied by the weights)
        # Replace by the new weights later
        # Don't normalize wgrid, if you normalize wgrid the data is not as smooth as it could be
        wgrid_values = np.nan_to_num(np.array(wgrid.values))
        grid_values = np.nan_to_num(np.array(grid.values))
    elif gridding_method == 'cvm':
        try:
            grid = omi.cvm_grid(grid, data['FoV75CornerLongitude'], data['FoV75CornerLatitude'],
                                values, errors, weights, missing_values)
        except QhullError as err:
            print("Cannot interpolate, QhullError: {0}".format(err.args[0]))
            return None, None

        wgrid_values = np.nan_to_num(grid.weights)
        grid.norm()
        grid_values = np.nan_to_num(grid.values)
    else:
        raise NotImplementedError('gridding method {0} not understood'.format(gridding_method))

    return grid_values, wgrid_values

def imatlab_gridding(data_in, grid_in, verbosity=0):
    # This is the interface function that should be called from Matlab to pass the Data structure as a list of
    # dictionaries. The conversion should happen on the Matlab side. This will verify that the required fields are
    # present in the dictionary
    omi.verbosity = verbosity - 1

    if isinstance(data_in, dict):
        test_data = data_in
        data_in = [data_in]
    elif isinstance(data_in, (list, tuple)):
        test_data = data_in[0]
        if any([not isinstance(x, dict) for x in data_in]):
            raise TypeError('If given as a list/tuple, all child elements of data_in must be dicts')
    else:
        raise TypeError('data_in must be a dict or a list/tuple of dicts')

    if __save_swath__:
        lastfile = sorted(glob('data-imatlab_gridding-pre_mask-*.he5'))
        if len(lastfile) > 0:
            lastfile = lastfile[-1]
            i_swath = int(re.search('\d\d', lastfile).group())
        else:
            i_swath = 0
        savename = 'data-imatlab_gridding-pre_mask-{:02d}.he5'.format(i_swath+1)
        print('Saving', savename)
        saveh5(savename, data=data_in[0])

    # Validate the fields present
    missing_fields = []
    field_type_warn = []
    for name in behr_datasets:
        if name not in test_data:
            missing_fields.append(name)
        elif type(test_data[name]) is not np.ndarray:
            s = '{0} {1}'.format(name, type(test_data[name]))
            field_type_warn.append(s)

    if len(missing_fields) > 0:
        raise KeyError('Required fields are missing: {}'.format(', '.join(missing_fields)))
    elif len(field_type_warn) > 0:
        print('WARNING: Some fields are not of type numpy.array:')
        print('\n'.join(field_type_warn))

    for swath in data_in:
        for k, v in swath.items():
            swath[k] = np.ma.masked_invalid(v)

    if __save_swath__:
        lastfile = sorted(glob('data-imatlab_gridding-post_mask-*.he5'))
        if len(lastfile) > 0:
            lastfile = lastfile[-1]
            i_swath = int(re.search('\d\d', lastfile).group())
        else:
            i_swath = 0
        savename = 'data-imatlab_gridding-post_mask-{:02d}.he5'.format(i_swath+1)
        print('Saving', savename)
        saveh5(savename, data=swath)

    return grid_day_from_interface(data_in, grid_in, 'psm', 'behr', verbosity=verbosity)

def main(verbosity=0):
    start = datetime.datetime(2013, 6, 1)
    stop = datetime.datetime(2013, 6, 2)
    #stop = datetime.datetime(2013, 8, 31)

    grid_info = {'llcrnrlat': 25.025, 'urcrnrlat': 50.0, 'llcrnrlon': -124.9750, 'urcrnrlon': -65.0, 'resolution': 0.05}
    #grid_info = {'llcrnrlat': 25.0, 'urcrnrlat': 50.05, 'llcrnrlon': -125.0, 'urcrnrlon': -64.95, 'resolution': 0.05}
    grid_method = 'psm'
    product = 'behr'

    #data_path = '/Volumes/share-sat/SAT/BEHR/WEBSITE/webData/PSM-Comparison/BEHR-PSM'  # where OMI-raw- data is saved
    data_path = '/Volumes/share-sat/SAT/BEHR/PSM_Tests_newprofiles_fixed_fills'
    #save_path = '/Users/Josh/Documents/MATLAB/BEHR/Workspaces/PSM-Comparison/Tests/UpdateBranch'  # path, where you want to save your data
    save_path = os.path.dirname(__file__)
    print('Will save to', save_path)

    #save_individual_days(start, stop, data_path, save_path, grid_info, grid_method, column_product=product, verbosity=verbosity)
    save_average(start, stop, data_path, save_path, grid_info, grid_method, column_product=product, verbosity=verbosity)

if __name__ == '__main__':
    omi.verbosity = __verbosity__ - 1
    main(verbosity=__verbosity__)