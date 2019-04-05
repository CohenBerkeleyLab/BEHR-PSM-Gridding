from __future__ import print_function

import datetime
from glob import glob
import numpy as np
import numpy.ma as ma
import os
import pdb
import re
#from scipy.spatial.qhull import QhullError

# This must be built and installed from the omi subdirectory. See omi/help.txt for instructions.
import omi

__author__ = 'Josh'
__verbosity__ = 1

# Define the datasets that need to be loaded from BEHR files
req_datasets = ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
                 'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude',
                 'CloudRadianceFraction', 'CloudPressure', 'BEHRColumnAmountNO2Trop', 'ColumnAmountNO2Trop',
                 'ColumnAmountNO2TropStd', 'BEHRAMFTrop', 'VcdQualityFlags', 'XTrackQualityFlags', 'SolarZenithAngle', 'Time',
                 'Latitude', 'Longitude']

# Define additional datasets required by the PSM gridding algorithm but not the CVM one
psm_req_datasets = ['SpacecraftAltitude', 'SpacecraftLatitude', 'SpacecraftLongitude']

# These are fields that, if gridded, should have their value masked for pixels affected by the row anomaly
row_anomaly_affected_fields = ['CloudFraction', 'CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2',
                               'SlantColumnAmountNO2', 'ColumnAmountNO2Trop', 'ColumnAmountNO2TropStd',
                               'ColumnAmountNO2Strat', 'BEHRColumnAmountNO2Trop', 'BEHRColumnAmountNO2TropVisOnly']

# These are fields that should be 1D because they have only the
# along or across track dimension (not both). We use this cell array during
# the conversion from Matlab arrays to Python Numpy arrays to represent
# this properly. This only matters in the rare case where only a single
# line of data is retained in an orbit (i.e. Longitude would be 1x60).
# This is important in two places in the gridding algorithm:
#   1) in generic_preprocessing, the area is replicated to be the same
#   shape as the quantity being gridded. If the gridded quantity is 1D,
#   this ends up making the area too big, because it uses the quantity
#   across track dimension as its along track dimension (since the along
#   track dimension doesn't exist)
#   2) in remove_out_of_domain_data (in omi.__init__) how it cuts down data
#   depends on if the data is 1, 2, or 3 dimensional. Therefore, it's
#   important to keep the actually 1D variables 1D so that it doesn't try
#   to use along and across track slicing on a 1D variable. (We could have
#   modified remove_out_of_domain_data to check if a 2D variable has a
#   length 1 dimension, but this way keeps the representation of 1D data on
#   the Python side consistent.)
one_d_fields = ['SpacecraftAltitude', 'SpacecraftLatitude', 'SpacecraftLongitude', 'Time', 'TiledArea', 'FoV75Area']

# These are fields that should be masked if the VcdQualityFlags field indicates a problem with the VCD algorithm
vcd_qualtity_affected_fields = ['ColumnAmountNO2Trop', 'ColumnAmountNO2TropStd', 'ColumnAmountNO2Strat',
                                'ColumnAmountNO2', 'BEHRColumnAmountNO2Trop', 'BEHRColumnAmountNO2TropVisOnly']

# Flag fields are fields that should be gridded using a bitwise OR operator rather than adding each value multiplied
# by its weight. This assumes that the flag uses 1 bits to indicate the presence of an error or warning during the
# process, so a grid cell resulting from multiple pixels should carry through any errors or warnings
flag_fields = ['VcdQualityFlags', 'XTrackQualityFlags', 'BEHRQualityFlags']

def behr_datasets(gridding_method='psm'):
    # Return the list of datasets required depending on the gridding method chosen.
    if gridding_method == 'psm':
        return req_datasets + psm_req_datasets
    else:
        return req_datasets

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

    return preprocessing(gridding_method=gridding_method, fov75_area=FoV75Area, gridded_quantity=BEHRColumnAmountNO2Trop,
                         gridded_quantity_std=ColumnAmountNO2TropStd, cloud_radiance_fraction=CloudRadianceFraction, mask=mask)

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

    return preprocessing(gridding_method=gridding_method, fov75_area=FoV75Area, gridded_quantity=ColumnAmountNO2Trop,
                         gridded_quantity_std=ColumnAmountNO2TropStd, cloud_radiance_fraction=CloudRadianceFraction, mask=mask)


def generic_preprocessing(gridding_method, gridded_quantity_name, gridded_quantity_values, FoV75Area, SolarZenithAngle,
                          VcdQualityFlags, XTrackQualityFlags, **kwargs):
    mask = gridded_quantity_values.mask
    #mask |= SolarZenithAngle > 85
    #if gridded_quantity_name in row_anomaly_affected_fields:
    #    mask |= XTrackQualityFlags != 0
    #if gridded_quantity_name in vcd_qualtity_affected_fields:
    #    mask |= VcdQualityFlags % 2 != 0

    return preprocessing(gridding_method, fov75_area=FoV75Area, gridded_quantity=gridded_quantity_values,
                         gridded_quantity_std=None, cloud_radiance_fraction=None, mask=mask)



def preprocessing(gridding_method, fov75_area, gridded_quantity, gridded_quantity_std, cloud_radiance_fraction, mask):
    """
    Subordinate preprocessing function that should be called from the specialized preprocessing function for individual
    columns. The specialized preprocessing function should create the mask to remove unwanted values (i.e. row anomaly
    or pixels with processing errors) and then pass the specified values to this function.

    :param gridding_method: the gridding method ('psm' or 'cvm') as a string
    :param gridded_quantity: the quantity to grid as a numpy array
    :param gridded_quantity_std: the standard error of the quantity to grid as a numpy array. If None is given, will be
        set to an array of ones the same size as gridded_quantity with the same data type and masking. This will
        effectively make the weights in CVM gridding just the inverse of the area.
    :param cloud_radiance_fraction: the cloud radiance fraction as a numpy array. If None is given, PSM gridding cannot
        be used.
    :param fov75_area: the FoV75 area as a numpy array; this should have a length of 1 in the along-track dimension and
        length 60 in the across track dimension (except maybe on zoom mode days)
    :param mask: a boolean array that is true where column values should not be used (e.g. row anomaly)

    Returns:
    :return:    values - masked array of BEHRColumnAmountNO2Trop values
                errors - masked array of ColumnAmountNO2TropStd values
                stddev - masked array of assumed standard deviations that increase linearly with Cloud Rad. Fraction
                weights - masked array of weights as the inverse of the pixel area (FoV75Area) if using PSM, or the
                    inverse of (pixel area * std dev.**2) for CVM. For PSM the weights also depend on the errors and
                    cloud radiance fraction.
    """
    if not isinstance(gridded_quantity, ma.core.MaskedArray):
        raise TypeError('gridded_quantity is expected to be a Masked Array, not {}'.format(type(gridded_quantity)))

    if not isinstance(fov75_area, ma.core.MaskedArray):
        raise TypeError('fov75_area is expected to be a Masked Array, not {}'.format(type(fov75_area)))

    if gridded_quantity_std is None:
        gridded_quantity_std = np.ones_like(gridded_quantity)
    elif not isinstance(gridded_quantity_std, ma.core.MaskedArray):
        raise TypeError('gridded_quantity_std is expected to be a Masked Array, not {}'.format(type(gridded_quantity_std)))

    # set invalid cloud cover to 100% -> smallest weight
    if cloud_radiance_fraction is None:
        pass
    elif isinstance(cloud_radiance_fraction, ma.core.MaskedArray):
        cloud_radiance_fraction[cloud_radiance_fraction.mask] = 1.0
    else:
        raise TypeError('cloud_radiance_fraction must be either None or a Masked Array, not {}'.format(type(cloud_radiance_fraction)))

    # VCD values and errors
    values = ma.array(gridded_quantity, mask=mask)
    errors = ma.array(gridded_quantity_std, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al. 2008)
    if cloud_radiance_fraction is not None:
        stddev = 1.5e15 * (1.0 + 3.0 * ma.array(cloud_radiance_fraction, mask=mask))
    else:
        stddev = gridded_quantity_std

    area = fov75_area.reshape((1, fov75_area.size))
    area = area.repeat(gridded_quantity.shape[0], axis=0)

    if gridding_method.lower() == 'psm':
        if cloud_radiance_fraction is None:
            # Strictly speaking, you *probably* can do PSM gridding without having cloud fraction - I don't think that
            # being able to use lower weights for cloudy pixels is necessary to make PSM work. That comes from Mark
            # Wenig's 2008 paper showing that the uncertainty increased with cloud fraction. However, I haven't thought
            # though how the weights should change if gridding a quantity that the uncertainty will not increase with
            # cloud fraction. JLL 9 Aug 2017
            raise NotImplementedError('Carrying out PSM gridding without cloud_radiance_fraction not implemented')

        weights = ma.array(1.0 / area, mask=mask)
        weights = ma.array(weights/((errors/1e16)*(1.3 + 0.87*cloud_radiance_fraction))**2, mask=mask)
    elif gridding_method.lower() == 'cvm':
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)
    else:
        raise NotImplementedError('No weighting formula specified for gridding method {}'.format(gridding_method))

    return values, errors, stddev, weights


def simple_preprocessing(values, errors, weights):
    """
    Preprocessing function for the simple interface to the CVM gridding code
    :param values: the values that will be gridded, as a regular numpy array
    :param errors: the errors for the values, as a regular numpy array
    :param weights: the weights for the values, as a regular numpy array
    :return: values, errors, weights as masked arrays with the same elements masked in all three, plus the mask itself
     The arrays will be masked by numpy.ma.masked_invalid
    """

    values = ma.masked_invalid(values)
    errors = ma.masked_invalid(errors)
    weights = ma.masked_invalid(weights)

    mask = values.mask | errors.mask | weights.mask

    values.mask = mask
    errors.mask = mask
    weights.mask = mask

    return values, errors, weights, mask


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



def grid_day_from_interface(behr_data, behr_grid, grid_method, gridded_quantity, preproc_method, verbosity=0):
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

    if not isinstance(gridded_quantity, str):
        raise TypeError('gridded_quantity must be a string')

    if not isinstance(preproc_method, str):
        raise TypeError('preproc_method must be a string')

    day_grid = make_grid(behr_grid)
    day_grid.zero()

    for data in behr_data:
        if verbosity > 0:
            print('Gridding swath {} of {}'.format(behr_data.index(data)+1, len(behr_data)))

        vals, weights = grid_orbit(data, behr_grid, gridded_quantity=gridded_quantity, gridding_method=grid_method,
                                   preproc_method=preproc_method, verbosity=verbosity)

        if vals is not None:
            day_grid.values += vals * weights
            day_grid.weights += weights

    day_grid.norm()
    return day_grid


def grid_orbit(data, grid_info, gridded_quantity, gridding_method='psm', preproc_method='generic', verbosity=0):
    # Input checking
    if not isinstance(data, dict):
        raise TypeError('data must be a dict')
    elif gridded_quantity not in data.keys():
        raise KeyError('data does not have a key matching the gridded_quantity "{}"'.format(gridded_quantity))
    else:
        missing_keys = [k for k in behr_datasets(gridding_method) if k not in data.keys()]
        if len(missing_keys) > 0:
            raise KeyError('data is missing the following expected keys: {0}'.format(', '.join(missing_keys)))

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
        print('    Doing {} preprocessing for {}'.format(preproc_method, gridded_quantity))
    if preproc_method.lower() == 'behr':
        values, errors, stddev, weights = behr_preprocessing(gridding_method, **data)
    elif preproc_method.lower() == 'sp':
        values, errors, stddev, weights = sp_preprocessing(gridding_method, **data)
    elif preproc_method.lower() == 'generic':
        # I'm copying the values before passing them because I intend this to be able to called multiple times to grid
        # different fields, and I don't want to risk values in data being altered.
        values, errors, stddev, weights = generic_preprocessing(gridding_method, gridded_quantity,
                                                                data[gridded_quantity].copy(), **data)
    else:
        raise NotImplementedError('No preprocessing option for column_product={}'.format(gridded_quantity))


    missing_values = values.mask.copy()

    if np.all(values.mask):
        return None, None  # two outputs expected, causes a "None object not iterable" error if only one given

    new_weight = weights # JLL 9 Aug 2017 - Seems unnecessary now



    if verbosity > 1:
        print('    Gridding {}'.format(gridded_quantity))

    is_flag_field = gridded_quantity in flag_fields

    if gridding_method == 'psm':
        gamma = omi.compute_smoothing_parameter(40.0, 40.0)
        rho_est = np.max(new_weight)*1.2
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

        rho_est = np.max(values)*1.2
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

        grid.norm()  # divides by the weights (at this point, the values in the grid are multiplied by the weights)
                     # Replace by new weights in a bit
        wgrid.norm()  # in the new version, wgrid is also normalized.

        # Clip the weights so that only positive weights are allowed. Converting things back to np.array removes any
        # masking
        wgrid.values = np.clip(np.array(wgrid.values), 0.01, np.max(np.array(wgrid.values)))

        wgrid_values = np.array(wgrid.values)
        grid_values = np.array(grid.values)
    elif gridding_method == 'cvm':
        try:
            grid = omi.cvm_grid(grid, data['FoV75CornerLongitude'], data['FoV75CornerLatitude'],
                                values, errors, weights, missing_values, is_flag=is_flag_field)
        except QhullError as err:
            print("Cannot interpolate, QhullError: {0}".format(err.args[0]))
            return None, None

        if not is_flag_field:
            wgrid_values = grid.weights
            grid.norm()
        else:
            wgrid_values = np.ones_like(grid.values)
        grid_values = grid.values
    else:
        raise NotImplementedError('gridding method {0} not understood'.format(gridding_method))

    # At this point, grid.values is a numpy array, not a masked array. Using nan_to_num should also automatically set
    # the value to 0, but that doesn't guarantee that the same values in weights will be made 0. So we manually multiply
    # the weights by 0 for any invalid values in the NO2 grid. This prevents reducing the final column when we divide by
    # the total weights outside this function.
    good_values = ~np.ma.masked_invalid(grid_values).mask
    grid_values = np.nan_to_num(grid_values)
    wgrid_values = np.nan_to_num(wgrid_values) * good_values
    return grid_values, wgrid_values

def imatlab_gridding(data_in, grid_in, field_to_grid, preprocessing_method='generic', gridding_method='psm', verbosity=0):
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

    # Validate the fields present
    missing_fields = []
    field_type_warn = []
    for name in behr_datasets(gridding_method):
        if name not in test_data:
            missing_fields.append(name)
        elif not isinstance(test_data[name], np.ndarray):
            s = '{0} {1}'.format(name, type(test_data[name]))
            field_type_warn.append(s)

    if len(missing_fields) > 0:
        raise KeyError('Required fields are missing: {}'.format(', '.join(missing_fields)))
    elif len(field_type_warn) > 0:
        print('WARNING: Some fields are not of type numpy.array:')
        print('\n'.join(field_type_warn))

    for swath in data_in:
        for k, v in swath.items():
            if k in one_d_fields:
                if v.ndim == 1:
                    continue
                    
                if verbosity > 2:
                    print('Squeezing field {} to 1D'.format(k))
                v = v.squeeze()
                # This is a kludge to avoid squeezing completely down to a 0D array since that breaks omi.remove_out_of_domain_data
                if v.ndim == 0:
                    v = v.reshape(1)
            swath[k] = np.ma.masked_invalid(v)

    return grid_day_from_interface(data_in, grid_in, gridding_method, field_to_grid, preprocessing_method,
                                   verbosity=verbosity)


def igridding_simple(grid_info, loncorn, latcorn, vals, errors=None, weights=None, is_flag=False):
    """
    Interface method that applies CVM gridding to a simple set of data defined in 2D numpy arrays
    :param grid_info: a dictionary that can be given to omi.Grid as omi.Grid(**grid_info)
    :param loncorn: the longitude corner coordinates of the data to grid as a 3D numpy array; assumes that the corners
     are given along the first dimension, i.e. loncorn[:,i,j] is the slice with the corner coordinates for pixel i, j
    :param latcorn: the latitude corner coordinate of the data to grid as a 3D numpy array; same format as loncorn.
    :param vals: the values to grid as a 2D numpy array
    :return: a 2D numpy array with the values gridded to the lon/lat coordinates defined in grid and a 2D numpy array
    with the weighting values
    """

    if errors is None:
        errors = np.zeros_like(vals)
    if weights is None:
        weights = np.ones_like(vals)

    if vals.ndim == 3:
        print('Doing 3D gridding')
        n_levels = vals.shape[2]
        for i_lev in range(n_levels):
            grid_val_i, grid_wt_i = igridding_simple(grid_info, loncorn, latcorn, vals[:, :, i_lev],
                                                     errors=errors[:, :, i_lev], weights=weights[:, :, i_lev],
                                                     is_flag=is_flag)
            if i_lev == 0:
                grid_values = np.full(grid_val_i.shape + (n_levels,), np.nan, dtype=grid_val_i.dtype)
                grid_weights = np.full(grid_wt_i.shape + (n_levels,), np.nan, dtype=grid_val_i.dtype)

            grid_values[:, :, i_lev] = grid_val_i
            grid_weights[:, :, i_lev] = grid_wt_i
    else:

        vals, errors, weights, mask = simple_preprocessing(vals, errors, weights)

        grid_in = omi.Grid(**grid_info)

        grid = omi.cvm_grid(grid_in, loncorn, latcorn, vals, errors, weights, mask, is_flag=is_flag)

        # For now, unlike the normal gridding method, we will retain NaNs in the gridded values and weights
        grid_weights = grid.weights
        grid.norm()
        grid_values = grid.values

    return grid_values, grid_weights


def main(verbosity=0):
    raise RuntimeError('Cannot run PSM_Main as primary program on the no-he5 branch')
    start = datetime.datetime(2013, 6, 1)
    #stop = datetime.datetime(2013, 6, 2)
    stop = datetime.datetime(2013, 8, 31)

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
    #save_average(start, stop, data_path, save_path, grid_info, grid_method, column_product=product, verbosity=verbosity)

if __name__ == '__main__':
    omi.verbosity = __verbosity__ - 1
    main(verbosity=__verbosity__)
