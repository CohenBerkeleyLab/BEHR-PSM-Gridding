from __future__ import print_function

import datetime
import numpy as np
import numpy.ma as ma
import os
import pdb
import re
from scipy.spatial.qhull import QhullError

# This must be built and installed from the omi subdirectory. See omi/help.txt for instructions.
import omi

__author__ = 'Josh'
__verbosity__ = 1



# Define the datasets that need to be loaded from BEHR files
behr_datasets = ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
                 'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude',
                 'Latitude', 'Longitude', 'SpacecraftAltitude', 'SpacecraftLatitude', 'SpacecraftLongitude',
                 'CloudRadianceFraction', 'CloudPressure', 'BEHRColumnAmountNO2Trop', 'ColumnAmountNO2TropStd',
                 'BEHRAMFTrop', 'vcdQualityFlags', 'XTrackQualityFlags', 'SolarZenithAngle', 'Time']


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



def behr_preprocessing(BEHRColumnAmountNO2Trop, ColumnAmountNO2TropStd, BEHRAMFTrop, SolarZenithAngle, vcdQualityFlags,
                       XTrackQualityFlags, CloudRadianceFraction, FoV75Area, **kwargs):
    """
    Preprocesses a dictionary of datasets from BEHR files. If the dictionary is "data", then it should be passed to this
    function using the ** syntax, i.e. preprocessing(**data). This will use the dictionary keys as keywords and the dict
    values as the values associated with those keywords. This helps keep the code here cleaner and makes clear what
    keys in the dictionary are required, which are:
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

    Returns:
    :return:    values - masked array of BEHRColumnAmountNO2Trop values
                errors - masked array of ColumnAmountNO2TropStd values
                stddev - masked array of assumed standard deviations that increase linearly with Cloud Rad. Fraction
                weights - masked array of weights as the inverse of the pixel area (FoV75Area)
    """

    # mask bad values - BEHR does not have its own std. deviation value. Very small AMFs typically indicate something
    # went wrong in the processing. This should probably be corrected in the BEHR algorithm itself
    amfmask = np.array(BEHRAMFTrop < 1e-5)
    cldmask = CloudRadianceFraction > 0.5
    mask = BEHRColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask | amfmask | cldmask

    # mask low quality data
    mask |= SolarZenithAngle > 85
    mask |= vcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags != 0

    # set invalid cloud cover to 100% -> smallest weight
    CloudRadianceFraction[CloudRadianceFraction.mask] = 1.0

    # VCD values and errors
    values = ma.array(BEHRColumnAmountNO2Trop, mask=mask)
    errors = ma.array(ColumnAmountNO2TropStd, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al. 2008)
    stddev = 1.5e15 * (1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    area = FoV75Area.reshape((1, FoV75Area.size))
    area = area.repeat(BEHRColumnAmountNO2Trop.shape[0], axis=0)

    weights = ma.array(1.0 / area, mask=mask)

    return values, errors, stddev, weights


def generate_filename(save_path, date_in, date_end=None):
    if date_end is None:
        fname = 'OMI_BEHR_PSM_{0}_{1}.he5'.format(behr_version(), date_in.strftime('%Y%m%d'))
    else:
        fname = 'OMI_BEHR_PSM_{0}_{1}-{2}.he5'.format(behr_version(),
                                                      date_in.strftime('%Y%m%d'),
                                                      date_end.strftime('%Y%m%d'))
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



def grid_day_from_file(behr_file, grid_info):
    day_grid = make_grid(grid_info)
    day_grid.zero() # ensure that values and weights are zeroed (just in case, Annette from Mark's group seems to be
                    # concerned that grids be zeroed out, though it seems like they really should start zeroed)

    for _, orbit_num, data in omi.he5.iter_behr_orbits_in_file(behr_file, behr_datasets):
        if __verbosity__ > 0:
            print(' Gridding orbit no. {0}'.format(orbit_num))
        vals, weights = grid_orbit(data, grid_info)
        if vals is not None:
            day_grid.values += np.nan_to_num(vals) * np.nan_to_num(weights)
            day_grid.weights += weights

    day_grid.norm()
    return day_grid


def save_individual_days(start_date, end_date, data_path, save_path, grid_info):
    for filename in omi.he5.iter_behr_filenames(start_date, end_date, data_path):
        if __verbosity__ > 0:
            print('Working on file {0}'.format(filename))
        day_grid = grid_day_from_file(filename, grid_info)
        mobj = re.search('\d\d\d\d\d\d\d\d', filename)
        behr_date = datetime.datetime.strptime(mobj.group(), '%Y%m%d')
        save_name = generate_filename(save_path, behr_date)
        day_grid.save_as_he5(save_name)


def multi_day_average(start_date, end_date, data_path, grid_info):
    avg_grid = make_grid(grid_info)
    avg_grid.zero()
    for filename in omi.he5.iter_behr_filenames(start_date, end_date, data_path):
        if __verbosity__ > 0:
            print('Working on file {0}'.format(filename))

        for _, orbit_num, data in omi.he5.iter_behr_orbits_in_file(filename, behr_datasets):
            if __verbosity__ > 0:
                print(' Gridding orbit no. {0}'.format(orbit_num))

            vals, weights = grid_orbit(data, grid_info)
            if vals is not None:
                avg_grid.values += np.nan_to_num(vals) * np.nan_to_num(weights)
                avg_grid.weights += weights

    avg_grid.norm()
    return avg_grid


def save_average(start_date, end_date, data_path, save_path, grid_info):
    avg = multi_day_average(start_date, end_date, data_path, grid_info)
    save_name = generate_filename(save_path, start_date, end_date)
    avg.save_as_he5(save_name)


def grid_orbit(data, grid_info):
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
    if __verbosity__ > 1:
        print('    Doing preprocessing')
    values, errors, stddev, weights = behr_preprocessing(**data)
    missing_values = values.mask.copy()

    if np.all(values.mask):
        return None, None  # two outputs expected, causes a "None object not iterable" error

    new_weight = weights / np.sqrt(
        (np.abs((errors / 1e15) * (1 + 2 * data['CloudRadianceFraction'] ** 2))))  # **(1.0/2.0)

    if __verbosity__ > 1:
        print('    Gridding VCDs')

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

    if __verbosity__ > 1:
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
    wgrid.values = np.nan_to_num(np.array(wgrid.values))
    grid.values = np.nan_to_num(np.array(grid.values))

    return grid.values.copy(), wgrid.values.copy()

def main():
    start = datetime.datetime(2015, 6, 1)
    stop = datetime.datetime(2015, 6, 2)
    #stop = datetime.datetime(2015, 8, 31)

    grid_info = {'llcrnrlat': 25.0, 'urcrnrlat': 50.05, 'llcrnrlon': -125.0, 'urcrnrlon': -64.95, 'resolution': 0.05}

    data_path = '/Volumes/share-sat/SAT/BEHR/WEBSITE/webData/PSM-Comparison/BEHR-PSM'  # where OMI-raw- data is saved
    save_path = '/Users/Josh/Documents/MATLAB/BEHR/Workspaces/PSM-Comparison/Tests/UpdateBranch'  # path, where you want to save your data

    save_individual_days(start, stop, data_path, save_path, grid_info)
    #save_average(start, stop, data_path, save_path, grid_info)

if __name__ == '__main__':
    omi.verbosity = __verbosity__ - 1
    main()