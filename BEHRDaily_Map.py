# /usr/bin/env python
# coding: utf-8

from datetime import datetime

import numpy as np
import numpy.ma as ma

import glob
#import sys
#sys.path.append('/usr/users/annette.schuett/Masterarbeit/omi-master/omi')
import omi
#import '/usr/users/annette.schuett/Masterarbeit/omi-master/omi'
import glob
from datetime import date
import calendar
import h5py
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import re
import pdb



#########################################################################
# This is an example script for gridding OMI data using the Python OMI 
# package (please start reading this file at the bottom after the
# "if __name__ == '__main__'" line.
#########################################################################



NAME2DATASET_PIXEL = {}
omi.he5.create_name2dataset(
    'Data/SWATHS*',
    ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
     'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],
    NAME2DATASET_PIXEL
)
omi.he5.create_name2dataset(
    'Data/SWATHS*',
    ['Latitude', 'Longitude', 'SpacecraftAltitude',
     'SpacecraftLatitude', 'SpacecraftLongitude'],
    NAME2DATASET_PIXEL
)

NAME2DATASET_NO2 = {}
omi.he5.create_name2dataset(
    'Data/SWATHS*',
    ['CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2Trop',
     'ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit',
     'VcdQualityFlags', 'XTrackQualityFlags'],
    NAME2DATASET_NO2
)
omi.he5.create_name2dataset(
    'Data/SWATHS*',
    ['SolarZenithAngle', 'Time'],
    NAME2DATASET_NO2
)
List = ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
     'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude',
     'Latitude', 'Longitude', 'SpacecraftAltitude',
     'SpacecraftLatitude', 'SpacecraftLongitude',
     'CloudRadianceFraction', 'CloudPressure', 'BEHRColumnAmountNO2Trop',
     'ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit',
     'vcdQualityFlags', 'XTrackQualityFlags',
     'SolarZenithAngle', 'Time']


def readin(Filename):
    f = h5py.File(Filename,'r')
    dictionary = {}
    #for name in f:
         #str = 'array'
         #str2 = str+'_%s' %np.str(name) # = np.get(name)
         #print str2
         #dictionary[str2] =  f.get(name)
         #print name
    return f  


def preprocessing(gridding_method, Time, BEHRColumnAmountNO2Trop,
    ColumnAmountNO2TropStd, FoV75Area, CloudRadianceFraction,
    RootMeanSquareErrorOfFit, SolarZenithAngle, vcdQualityFlags,
    XTrackQualityFlags, **kwargs):

    # mask of bad values
    mask = BEHRColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask

    # mask low quality data
    mask |= RootMeanSquareErrorOfFit > 0.0003
    mask |= SolarZenithAngle > 85
    mask |= vcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags != 0
    mask |= CloudRadianceFraction > 0.5

    # set invalid cloud cover to 100% -> smallest weight
    CloudRadianceFraction[CloudRadianceFraction.mask] = 1.0

    # values and errors
    values = ma.array(BEHRColumnAmountNO2Trop, mask=mask)
    errors = ma.array(ColumnAmountNO2TropStd, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al., 2008)
    stddev = 1.5e15 * (1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    area = FoV75Area.reshape(1, FoV75Area.size)
    area = area.repeat(BEHRColumnAmountNO2Trop.shape[0], axis=0)

    
    weights = ma.array(1.0 / area, mask=mask)
    

    return values, errors, stddev, weights


def main(year, month, day, gridding_method, grid_name, List):

    # 1. Define a grid
    # (a) by giving lower-left and upper-right corner
    grid_name = "northamerica_josh"
    #grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.002); grid_name = 'Germany'#7500*12500
    #grid = omi.Grid(llcrnrlat= 17.8 , urcrnrlat=53.6 ,llcrnrlon=96.9 , urcrnrlon= 106.8, resolution=0.01); #grid_name = 'Northamerica'#6000*4000
    grid = omi.Grid.by_name(grid_name)
    # (b) or by reading this data from a JSON file
    #    (the default file can be found in omi/data/gridds.json)
   # grid = omi.Grid.by_name(grid_name)

    # 2. Define parameter for PSM
    #    - gamma (smoothing parameter)
    #    - rho_est (typical maximum value of distribution)
    rho_est = 4e16
    if gridding_method == 'psm':
        # gamma is computed as function of pixel overlap
        gamma = omi.compute_smoothing_parameter(1.0, 10.0)

    # 3. Define a mapping which maps a key to the path in the
    #    HDF file. The function
    #    >>> omi.he5.create_name2dataset(path, list_of_dataset_names, dict)
    #    can be helpful (see above).
    name2datasets = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]
    
    filename = '/usr/users/annette.schuett/Masterarbeit/Vergleich_Ber/HDF/BEHR-PSM/OMI_BEHR_v2-1B_%s%s%s.hdf'%(year, month, day)
    print filename
    
    #f = readin(filename)  
    f = h5py.File(filename, 'r')
    data = dict()
    """
    for orbit in f['Data']:
        for name in f['Data'][orbit]:
            print orbit, name
            value = f['Data'][orbit][name]
            
            if value.dtype == np.float32:
                data[name] = ma.array(value)#, mask=mask, dtype=np.float64)
            else:
                data[name] = ma.array(value)
                
    """
    
    for orbit in f['Data']:
        #for name in f['Data'][orbit]:
        for i in range(len(List)):
            name = List[i]
            #print name,':'
            field = f.get(('Data/%s/%s')%(orbit,name), None)
            if field is None:
                print name, 'Warning'#: No data of name.' % (name)
                data[name] = None
            else: 
                #print '----------------------------------------<worked'
                
                #fill_value = field.attrs.get('_FillValue', None)
                fill_value = field.fillvalue
                print name, fill_value
                #missing_value = field.attrs.get('MissingValue', None)
                missing_value = fill_value
                scale = field.attrs.get('ScaleFactor', 1.0)
                offset = field.attrs.get('Offset', 0.0)

                if fill_value is None and missing_value is None:
                    raise ValueError('Missing `_FillValue` or `MissingValue` at orbit %s wirth product %s ' % (orbit, name))

                value = field.value
                mask = (field.value == fill_value) | (field.value == missing_value)

                if fill_value < -1e25 or missing_value < -1e25:
                    mask |= (field.value < -1e25)

                if abs(scale - 1.0) > 1e-12:
                    value = value * scale

                if abs(offset - 0.0) > 1e-12:
                    value = value + offset

                if value.dtype == np.float32:
                    data[name] = ma.array(value, mask=mask, dtype=np.float64)
                else:
                    data[name] = ma.array(value, mask=mask, dtype=value.dtype)
                if len(np.shape(data[name]))==2:
                    data[name] = data[name].T
                
                    
                print name, np.shape(data[name])    
                    
                    
                                
        ##data['TiledCornerLongitude'] = np.transpose(data['TiledCornerLongitude'], axes =(1,0,2))
        ##data['TiledCornerLatitude'] = np.transpose(data['TiledCornerLatitude'], axes =(1,0,2))
        data['TiledCornerLongitude'] = np.transpose(data['TiledCornerLongitude'], axes =(2,1,0))
        data['TiledCornerLatitude'] = np.transpose(data['TiledCornerLatitude'], axes =(2,1,0))
        data['FoV75CornerLatitude'] = np.transpose(data['FoV75CornerLatitude'], axes =(2,1,0))
        data['FoV75CornerLongitude'] = np.transpose(data['FoV75CornerLongitude'], axes =(2,1,0))
        
        
        data['TiledArea'] = data['TiledArea'].T[0]
        data['FoV75Area'] = data['FoV75Area'].T[0]
        data['SpacecraftAltitude'] = data['SpacecraftAltitude'].T[0]
        data['SpacecraftLatitude'] = data['SpacecraftLatitude'].T[0]
        data['SpacecraftLongitude'] = data['SpacecraftLongitude'].T[0]
        data['Time'] = data['Time'].T[0]
        print np.shape(data['TiledCornerLongitude']), np.shape(data['FoV75Area'])
        for i in range(len(List)):
            print List[i],':', np.shape(data[List[i]])
            
            
            
        print ''    
        
        
        # 5) Check for missing corner coordinates, i.e. the zoom product,
        #    which is currently not supported
        if (data['TiledCornerLongitude'].mask.any() or
            data['TiledCornerLatitude'].mask.any()
        ):
            continue

        # 6) Clip orbit to grid domain
        lon = data['FoV75CornerLongitude']
        lat = data['FoV75CornerLatitude']
        #pdb.set_trace()
        data = omi.clip_orbit(grid, lon, lat, data, boundary=(2,2))
    

        if data['BEHRColumnAmountNO2Trop'].size == 0:
            continue

        # 7) Use a self-written function to preprocess the OMI data and
        #    to create the following arrays MxN:
        #    - measurement values 
        #    - measurement errors (currently only CVM grids errors)
        #    - estimate of stddev (used in PSM)
        #    - weight of each measurement
        #    (see the function preprocessing for an example)
        values, errors, stddev, weights = preprocessing(gridding_method, **data)
        missing_values = values.mask.copy()

        if np.all(values.mask):
            continue


        # 8) Grid orbit using PSM or CVM:
        #print 'time: %s, orbit: %d' % (timestamp, orbit)
        if gridding_method == 'psm':
            grid = omi.psm_grid(grid,
                data['Longitude'], data['Latitude'],
                data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                values, errors, stddev, weights, missing_values,
                data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                data['SpacecraftAltitude'],
                gamma[data['ColumnIndices']],
                rho_est
            )
        else:
            grid = omi.cvm_grid(grid, data['FoV75CornerLongitude'], data['FoV75CornerLatitude'],
            values, errors, weights, missing_values)


    # 9) The distribution of values and errors has to be normalised
    #    with the weight.
    grid.norm()

    # 10) The Level 3 product can be saved as HDF5 file
    #     or converted to an image (requires matplotlib and basemap
    grid.save_as_he5('%s_%s_%s_%s_%s.he5' % (grid_name,year,  month,  day, gridding_method))
    grid.save_as_image('%s_%s_%s_%s_%s.png' % (grid_name, year,  month,  day, gridding_method), vmin=0, vmax=rho_est)
    #grid.save_as_image('%s_%s_%s_%s_%s.he5' % ( str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4], grid_name, gridding_method), vmin=0, vmax=rho_est)

    # 11) It is possible to set values, errors and weights to zero.
    grid.zero()
    



if __name__ == '__main__':


    grid_name = 'NA'

    
    
    files = glob.glob('/usr/users/annette.schuett/Masterarbeit/Vergleich_Ber/HDF/BEHR-PSM/OMI_BEHR_v2-1B_2015*.hdf')
    np.sort(files)
    counter = 0
    
    
    for filename in files:
        
        year = filename[82:86]
        month = filename[86:88]
        day = filename[88:90]
        main(year, month, day, 'psm', grid_name, List)
        counter +=1
        #if counter == 2:
            #break
        #print year, month, day   
    
    
        #start_date = datetime(year,month,day)
        #end_date = datetime(year,month,day+1)
    """
    print "start_date =", start_date 

    # Call main function twice to grid data using CVM and PSM
    main(start_date, end_date, 'cvm', grid_name, data_path)
   
    print "First Algorithm ready"
     """
        #main(year, month, day, 'psm', grid_name, data_path)
  






