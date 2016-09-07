# /usr/bin/env python
# coding: utf-8

from datetime import datetime

import numpy as np
import numpy.ma as ma

import omi

import multiprocessing
import os


#########################################################################
# This is an example script for gridding OMI data using the Python OMI 
# package (please start reading this file at the bottom after the
# "if __name__ == '__main__'" line.
#########################################################################



NAME2DATASET_PIXEL = {}
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields',
    ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
     'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],
    NAME2DATASET_PIXEL
)
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields',
    ['Latitude', 'Longitude', 'SpacecraftAltitude',
     'SpacecraftLatitude', 'SpacecraftLongitude'],
    NAME2DATASET_PIXEL
)

NAME2DATASET_NO2 = {}
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields',
    ['CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2Trop',
     'ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit',
     'VcdQualityFlags', 'XTrackQualityFlags'],
    NAME2DATASET_NO2
)
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields',
    ['SolarZenithAngle', 'Time'],
    NAME2DATASET_NO2
)



def preprocessing(gridding_method, Time, ColumnAmountNO2Trop,
    ColumnAmountNO2TropStd, FoV75Area, CloudRadianceFraction,
    RootMeanSquareErrorOfFit, SolarZenithAngle, VcdQualityFlags,
    XTrackQualityFlags, **kwargs):

    # mask of bad values
    mask = ColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask

    # mask low quality data
    mask |= RootMeanSquareErrorOfFit > 0.0003
    mask |= SolarZenithAngle > 85
    mask |= VcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags != 0

    # set invalid cloud cover to 100% -> smallest weight
    CloudRadianceFraction[CloudRadianceFraction.mask] = 1.0

    # values and errors
    values = ma.array(ColumnAmountNO2Trop, mask=mask)
    errors = ma.array(ColumnAmountNO2TropStd, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al., 2008)
    stddev = 1.5e15 * (1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    area = FoV75Area.reshape(1, FoV75Area.size)
    area = area.repeat(ColumnAmountNO2Trop.shape[0], axis=0)

    if gridding_method.startswith('psm'):
        weights = ma.array(1.0 / area, mask=mask)
    else:
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)

    return values, errors, stddev, weights




def main(start_date):
    #print start_date, type(start_date)
    grid_name  = 'europe'
    try:
        gridding_method = 'psm'
        start_date_str = start_date.strftime("%Y-%m-%d")
        year = int(start_date_str[0:4])
        month = int(start_date_str[5:7]) 
        day = int(start_date_str[8:10])
        grid = omi.Grid(llcrnrlat=30.0, urcrnrlat=46.0,llcrnrlon= 124.0, urcrnrlon= 146.0, resolution=0.004); grid_name = 'Japan'#7500*12500
        
        if day == lenghtmonth(year, month):
            day2 = 1
            month2 = month +1
            year2=year
            if month == 12:
                if day == 31:
                    year2 = year+1
                    month2 = 1
        else:
            day2 = day+1
            month2 = month
            year2=year

        end_date = datetime(year2,month2,day2)
            
        name = '/usr/users/annette.schuett/Masterarbeit/omi-master/Japan/%s_%s_%s_%s_%s.he5' % (grid_name, str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4], gridding_method)
        #print name
        n = 0

        if os.path.isfile(name)== True:
            if os.stat(name).st_size >10e5:
                status = 'File_Exist'
                #print name
                #print status
                n = n+1
            else:
                status = 'Do_again'
                #print status, ", the following file is not existing: ", name
        else:
            status = 'Do_again'

        
        
        
        if status == 'Do_again':
            print status, name
    
            
            data_path = '/home/zoidberg/OMI'
            datapath = data_path
            
            
            start_date_str = start_date.strftime("%Y-%m-%d")
            year = int(start_date_str[0:4])
            month = int(start_date_str[5:7]) 
            day = int(start_date_str[8:10])
            
            if day == lenghtmonth(year, month):
                day2 = 1
                month2 = month +1
                year2=year
                if month == 12:
                    if day == 31:
                        year2 = year+1
                        month2 = 1
            else:
                day2 = day+1
                month2 = month
                year2=year

            end_date = datetime(year2,month2,day2)
            




            # 1. Define a grid
            # (a) by giving lower-left and upper-right corner
            
            #grid = omi.Grid(llcrnrlat=-60.0, urcrnrlat=20.0, llcrnrlon=-83.0, urcrnrlon=-32.0, resolution=0.1); grid_name = 'South_America'
            
            # (b) or by reading this data from a JSON file
            #    (the default file can be found in omi/data/gridds.json)
            #grid = omi.Grid.by_name(grid_name)

            # 2. Define parameter for PSM
            #    - gamma (smoothing parameter)
            #    - rho_est (typical maximum value of distribution)
            rho_est = 4e15   # !!!!!!!!!!!!! Normalerweise: 4e16
            gridding_method = 'psm'
                # gamma is computed as function of pixel overlap
            gamma = omi.compute_smoothing_parameter(1.0, 10.0)

            # 3. Define a mapping which maps a key to the path in the
            #    HDF file. The function
            #    >>> omi.he5.create_name2dataset(path, list_of_dataset_names, dict)
            #    can be helpful (see above).
            name2datasets = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]

            # 4a) data in OMI files can be read by
            # >>> data = omi.he5.read_datasets(filename, name2dataset)

            # 4b) or by iterating over orbits from start to end date at the following
            #   location: 
            #       os.path.join(data_path, product, 'level2', year, doy, '*.he5')
            #
            #   (see omi.he5 module for details)
            products = ['OMNO2.003', 'OMPIXCOR.003']
            for timestamp, orbit, data in omi.he5.iter_orbits(
                    start_date, end_date, products, name2datasets, data_path
                ):

                # 5) Check for missing corner coordinates, i.e. the zoom product,
                #    which is currently not supported
                if (data['TiledCornerLongitude'].mask.any() or
                    data['TiledCornerLatitude'].mask.any()
                ):
                    continue

                # 6) Clip orbit to grid domain
                lon = data['FoV75CornerLongitude']
                lat = data['FoV75CornerLatitude']
                data = omi.clip_orbit(grid, lon, lat, data, boundary=(2,2))

                if data['ColumnAmountNO2Trop'].size == 0:
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
                print 'time: %s, orbit: %d' % (timestamp, orbit)

                grid = omi.psm_grid(grid,
                    data['Longitude'], data['Latitude'],
                    data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                    values, errors, stddev, weights, missing_values,
                    data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                    data['SpacecraftAltitude'],
                    gamma[data['ColumnIndices']],
                    rho_est
                )

            # 9) The distribution of values and errors has to be normalised
            #    with the weight.
            grid.norm()

            # 10) The Level 3 product can be saved as HDF5 file
            #     or converted to an image (requires matplotlib and basemap
            grid.save_as_he5('/usr/users/annette.schuett/Masterarbeit/omi-master/Japan/%s_%s_%s_%s_%s.he5' % (grid_name, str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4], gridding_method))
            #grid.save_as_image('/home/zoidberg/OMI/Germay/%s_%s_%s_%s_%s.png' % (grid_name, str(start_date)[8:10],str(start_date)[5:7],  str(start_date)[0:4], gridding_method), vmin=0, vmax=rho_est)
            #grid.save_as_image('%s_%s_%s_%s_%s.he5' % ( str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4], grid_name, gridding_method), vmin=0, vmax=rho_est)

            # 11) It is possible to set values, errors and weights to zero.
            grid.zero()
    except:
        print "Oh je, da funktioniert der Algorithmus noch nicht: Noch Mal anschauen: Monat", month, "den", day, ".ten Tag"            

    #Unterscheidung Schaltjahr, andere Jahre
    
def lenghtmonth(year, month):
    if year%4 == 0:
        lastday = {"1" : 31, "2" : 29, "3" : 31, "4" : 30, "5" : 31, "6" : 30, "7" : 31, "8" : 31, "9" : 30, "10" : 31, "11" : 30, "12" : 31 }
    else:
        lastday = {"1" : 31, "2" : 28, "3" : 31, "4" : 30, "5" : 31, "6" : 30, "7" : 31, "8" : 31, "9" : 30, "10" : 31, "11" : 30, "12" : 31 }
    return lastday[str(month)]  
    
def generating_start_end(startyear, endyear, algorithmname, gridname, datapath):
    start = []
    end =[]
    grid_name = []
    data_path = []
    algorithm = []
    for year in range(startyear, endyear+1):
        for month in range(1,13):
            for day in range(1, lenghtmonth(year, month)+1):
                start_date = datetime(year,month,day)
                if day == lenghtmonth(year, month):
                    day2 = 1
                    month2 = month +1
                    year2=year
                    if month == 12:
                        if day == 31:
                            year2 = year+1
                            month2 = 1
                else:
                    day2 = day+1
                    month2 = month
                    year2=year
                
                #print year, month, day, year2, month2, day2
                start_date = datetime(year,month,day)
                end_date = datetime(year2,month2,day2)
                start.append(start_date)
                end.append(end_date)
                grid_name.append(gridname)
                data_path.append(datapath)
                algorithm.append(algorithmname)
    return start, end, algorithm, grid_name, data_path     
    



def generating_start(startyear, endyear):
    start = []
    for year in range(startyear, endyear+1):
        for month in range(1,13):
            for day in range(1, lenghtmonth(year, month)+1):
                start_date = datetime(year,month,day)
                start.append(start_date)
    return start
    




if __name__ == '__main__':
    
    #multiprocessing, define map 
    parallel = True # False: Nur ein Kern, True: 8 oder wie viele auch immer Kerne
    if parallel:
        from multiprocessing import Pool
        #pool = multiprocessing.Pool(6)
        pool = Pool(1)
        mymap = pool.map
    else:
        mymap = map
    
    startyear = 2008
    endyear = 2015  # bereits fertiggestellt: 2005, 2006

    # "data_path" is the root path to your OMI data. Please change it for
    # you settings. 
    #
    # The OMI files are assumed to be location at:
    #    "{data_path}/{product}/level2/{year}/{doy}/*.he5"
    #
    # For example:
    #    "/home/gerrit/Data/OMI/OMNO2.003/level2/2006/123/*.he5"
    # or
    #    "/home/gerrit/Data/OMI/OMNO2.003/level2/2006/123/*.he5"
    #data_path = '/home/gerrit/Data/OMI'

    
    
    
    


    start = generating_start(startyear, endyear)
    
    pool.map(main, start)
 
    #algorithmname = 'psm'
    #mymap(lambda (start, end,algorithm , grid_name, data_path): main(start, end,algorithm , grid_name, data_path), zip(start, end,algorithm ,grid_name, data_path))
    
    
    """
    for year in range(2004, 2016):
        for month in range(1,12):
            for day in range(1, lastday[str(month)]+1):
                #print "month=", month, ", day =", day
                start_date = datetime(year,month,day)
                if day == lastday[str(month)]:
                    day2 = 1
                    month = month +1
                    year2 =year
                else:
                    day2 = day +1
                    month2= month
                    year2 = year
                end_date = datetime(year,month,day2)
                #if month ==12:
                    #if day== 31:
                        #year2= year+1
                        #month2=1
                        #day2=1
            
                # Call main function twice to grid data using CVM and PSM
                print "year =", year, ",month =", month, ",day:", day, ",", "year2 =", year2, ",month2 =", month2, ",day2:", day2
                #main(start_date, end_date, 'cvm', grid_name, data_path)
                #print "First Algorithm ready"
                #main(start_date, end_date, 'psm', grid_name, data_path)
            """

    """
    start_date = datetime(year,month,day)
    end_date = datetime(year,month,day+1)

    print "start_date =", start_date 

    # Call main function twice to grid data using CVM and PSM
    main(start_date, end_date, 'cvm', grid_name, data_path)
    
    print "First Algorithm ready"
    
    main(start_date, end_date, 'psm', grid_name, data_path)
    """























