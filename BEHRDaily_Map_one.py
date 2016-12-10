# Author: Annette Schuett, Gerret Kuhlmann


# /usr/bin/env python
# coding: utf-8

import datetime

import numpy as np
import numpy.ma as ma
from multiprocessing import Pool
import os.path
import pdb


#import sys
#sys.path.append('/usr/users/annette.schuett/Masterarbeit/omi-master/omi')
import omi

#import '/usr/users/annette.schuett/Masterarbeit/omi-master/omi'

#      new programm, without mask |= RootMeanSquareErrorOfFit > 0.0003, because the new product has no RootMeanSquareErrorOfFit




#########################################################################
# This is an example script for gridding OMI data using the Python OMI 
# package (please start reading this file at the bottom after the
# "if __name__ == '__main__'" line.
#########################################################################



NAME2DATASET_PIXEL = {}
omi.he5.create_name2dataset(
    '/Data/SWATHS*',
    ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
     'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],
    NAME2DATASET_PIXEL
)
omi.he5.create_name2dataset(
    '/Data/SWATHS*',
    ['Latitude', 'Longitude', 'SpacecraftAltitude',
     'SpacecraftLatitude', 'SpacecraftLongitude'],
    NAME2DATASET_PIXEL
)

NAME2DATASET_NO2 = {}
omi.he5.create_name2dataset(
    '/Data/SWATHS*',
    ['CloudRadianceFraction', 'CloudPressure', 'BEHRColumnAmountNO2Trop',
     'ColumnAmountNO2TropStd',
     'VcdQualityFlags', 'XTrackQualityFlags'],
    NAME2DATASET_NO2
)
omi.he5.create_name2dataset(
    '/Data/SWATHS*',
    ['SolarZenithAngle', 'Time'],
    NAME2DATASET_NO2
)



def preprocessing(Time, ColumnAmountNO2Trop,
    ColumnAmountNO2TropStd, FoV75Area, CloudRadianceFraction, 
    SolarZenithAngle, VcdQualityFlags,
    XTrackQualityFlags, **kwargs):

    # mask of bad values
    mask = ColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask

    # mask low quality data
    #mask |= RootMeanSquareErrorOfFit > 0.0003
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

    #if gridding_method.startswith('psm'):
    weights = ma.array(1.0 / area, mask=mask)
    #else:
        #weights = ma.array(1.0 / (area * stddev**2), mask=mask)

    return values, errors, stddev, weights


def generate_zip_lists(start, stop, grid_name, data_path, save_path):
    startlist = []
    stoplist = []
    gridlist = []
    pathlist = []
    savelist = []
    daydate = start 
    while daydate != stop:
        startlist.append(daydate)
        daydate += datetime.timedelta(days = 1)
        stoplist.append(daydate)
        gridlist.append(grid_name)
        pathlist.append(data_path)
        savelist.append(save_path)
    
    return startlist, stoplist, gridlist, pathlist, savelist

def gridname2grid(grid_name): #return grid
    #pdb.set_trace()
    if grid_name=='Global':
        grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.05);# grid_name = 'Global' # 3600 x 7200
        
    elif grid_name=='MiddleEurope':
        grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.004);# grid_name = 'MiddleEurope' # 3750 x 6250
    elif grid_name=='Europe':
        grid = omi.Grid(llcrnrlat=35.0, urcrnrlat=60.0,llcrnrlon=-10.0, urcrnrlon=25.0, resolution=0.005);# grid_name = 'Europe' # 5000 x 7000
    elif grid_name=='Austria':
        grid = omi.Grid(llcrnrlat= 46.2, urcrnrlat= 49.2,llcrnrlon= 9.3, urcrnrlon= 17.3, resolution= 0.001);# grid_name = 'Austria' # 3000 x 8000 
    elif grid_name=='Hungary':
        grid = omi.Grid(llcrnrlat= 45.5, urcrnrlat= 49.0,llcrnrlon= 15.7, urcrnrlon= 23.3, resolution=0.001);# grid_name = 'Hungary' # 3500 * 7600
    elif grid_name=='France':
        grid = omi.Grid(llcrnrlat=41.0, urcrnrlat= 52.0,llcrnrlon= -5.2, urcrnrlon= 8.9, resolution=0.002);# grid_name = 'France' # 5500 * 7050    
    elif grid_name=='Germany':
        grid = omi.Grid(llcrnrlat=45.5, urcrnrlat= 55.5,llcrnrlon= 5.0, urcrnrlon= 17.5, resolution=0.002);# grid_name = 'Germany' # 5000 * 6250  
        
    elif grid_name=='Northamerica':
        grid = omi.Grid(llcrnrlat=15.0, urcrnrlat=55.0,llcrnrlon=-125.0, urcrnrlon=-65.0, resolution=0.01); #grid_name = 'Northamerica' # 4000 * 6000
    elif grid_name=='Southamerica':
        grid = omi.Grid(llcrnrlat=-60.0, urcrnrlat=20.0,llcrnrlon= -83.0, urcrnrlon= -32.0, resolution=0.01); #grid_name = 'Southamerica' # 8000 * 5100
    elif grid_name=='Brasilia':
        grid = omi.Grid(llcrnrlat= -34.6, urcrnrlat= 9.4,llcrnrlon= -76.6, urcrnrlon= -33.2, resolution=0.008); #grid_name = 'Brasilia' # 5500 * 5425
        
    elif grid_name=='Asia':
        grid = omi.Grid(llcrnrlat=11.0, urcrnrlat=55.0,llcrnrlon=70.0, urcrnrlon=102.0, resolution=0.008); #grid_name = 'Asia' # 5500 * 4000
    elif grid_name=='PearlRiverDelta':
        grid = omi.Grid(llcrnrlat= 19.6, urcrnrlat= 25.6,llcrnrlon= 108.9, urcrnrlon= 117.6, resolution=0.0015); #grid_name = 'PearlRiverDelta' # 4000 * 5800
    elif grid_name=='Industrial_China':
        grid = omi.Grid(llcrnrlat= 20.5, urcrnrlat= 42.0,llcrnrlon= 102.0, urcrnrlon= 125.0, resolution=0.004);# grid_name = 'Industrial_China' # 5375 * 5750
    elif grid_name=='Korea_Japan':
        grid = omi.Grid(llcrnrlat= 30.0, urcrnrlat= 46,llcrnrlon= 124.0, urcrnrlon= 146.0, resolution=0.004);# grid_name = 'Korea_Japan' # 4000 * 5500
    elif grid_name=='Shanghai':
        grid = omi.Grid(llcrnrlat= 30.0, urcrnrlat= 33.0,llcrnrlon= 119.0, urcrnrlon= 123.0, resolution=0.001);# grid_name = 'Shanghai' # 3000 * 4000       #0006    
    
    elif grid_name=='MiddleEast':
        grid = omi.Grid(llcrnrlat= 11.8, urcrnrlat= 46.0,llcrnrlon= 25.4, urcrnrlon= 77.0, resolution=0.008); #grid_name = 'MiddleEast' # 4275 * 6450
    elif grid_name=='Syria':
        grid = omi.Grid(llcrnrlat= 32.05, urcrnrlat= 37.55,llcrnrlon= 35.5, urcrnrlon= 42.5, resolution=0.001); #grid_name = 'Syria' # 5500 * 7000    
    elif grid_name=='Israel':
        grid = omi.Grid(llcrnrlat= 29.0, urcrnrlat= 33.5,llcrnrlon= 33.9, urcrnrlon= 36.9, resolution=0.001);# grid_name = 'Israel' # 7500 * 5000, res von 0.0006 auf 0.001
    
        
        
    elif grid_name=='Australia':
        grid = omi.Grid(llcrnrlat= -45.0, urcrnrlat= -9.0,llcrnrlon= 110.0, urcrnrlon= 155.0, resolution=0.006); #grid_name = 'Australia' # 6000 * 7500 
    elif grid_name=='NewZealand':
        grid = omi.Grid(llcrnrlat= -53.0, urcrnrlat= -32.0,llcrnrlon= 165.0, urcrnrlon= 180.0, resolution=0.003); #grid_name = 'NewZealand' # 7000 * 5000
    
    
    elif grid_name=='Africa':
        grid = omi.Grid(llcrnrlat= -36.0, urcrnrlat= 40.0,llcrnrlon= -19.0, urcrnrlon= 52.0, resolution=0.01); #grid_name = 'Africa' # 7600 * 7100

    elif grid_name=='NorthAmericaBEHR':
        grid = omi.Grid(llcrnrlat= 25.0, urcrnrlat= 50.05, llcrnrlon= -125.0, urcrnrlon= -64.95, resolution= 0.05)

    return grid


def generate_filename(save_path, startdate,grid_name):

    
    grid=gridname2grid(grid_name)
    filename='%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s)'% (save_path, grid_name, startdate.year, startdate.month, startdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    
    return filename






def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

#@unpack_args
def main(start_date, end_date, grid_name, data_path, save_path):
    
    
    

    # 1. Define a grid

    #grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.002); grid_name = 'Germany'#7500*12500 
    #grid = omi.Grid.by_name(grid_name)
    #grid = gridname2grid(grid_name)
    
    grid = gridname2grid(grid_name)
    gridcoll = gridname2grid(grid_name)
    wgrid = gridname2grid(grid_name)
    
    grid.values = grid.values * 0.0
    wgrid.values = wgrid.values * 0.0
    gridcoll.values = gridcoll.values * 0.0
    gridcoll.weights = gridcoll.weights * 0.0
    
    
    
    filename  = generate_filename(save_path, start_date,grid_name)
    
    
    fname = '%s.he5' % (filename)
    
    if os.path.isfile(fname) == True:
        print('Existing file:         ',  fname)
    
    
    else:
            

        try:
            

            

            # 2. Define parameter for PSM
            #    - gamma (smoothing parameter)
            #    - rho_est (typical maximum value of distribution)
            rho_est = 4e16
            
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
            #products = ['OMNO2.003', 'OMPIXCOR.003']
            products = ['BEHR-PSM', 'BEHR-PSM']
            pdb.set_trace()
            for timestamp, orbit, data in omi.he5.iter_orbits(
                    start_date, end_date, products, name2datasets, data_path
                ):
                print('time: %s, orbit: %d' % (timestamp, orbit))
                grid = gridname2grid(grid_name)
                wgrid = gridname2grid(grid_name)
                
                #print '1'
                

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
                
                
                #print '2'
                

                # 7) Use a self-written function to preprocess the OMI data and
                #    to create the following arrays MxN:
                #    - measurement values 
                #    - measurement errors (currently only CVM grids errors)
                #    - estimate of stddev (used in PSM)
                #    - weight of each measurement
                #    (see the function preprocessing for an example)
                values, errors, stddev, weights = preprocessing(**data)
                missing_values = values.mask.copy()

                if np.all(values.mask):
                    continue
                
                
                #new_weight = 1/np.sqrt(np.abs((errors/1e15) * (1+2*data['CloudRadianceFraction']**2)))#**(1.0/2.0)
                new_weight = weights/np.sqrt((np.abs((errors/1e15) * (1+2*data['CloudRadianceFraction']**2))))#**(1.0/2.0)


                
                #print 'time: %s, orbit: %d' % (timestamp, orbit)
                #print '-----------------------------'
            
                rho_est = 4e16
                gamma = omi.compute_smoothing_parameter(1.0, 10.0)
                grid = omi.psm_grid(grid,
                    data['Longitude'], data['Latitude'],
                    data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                    values, errors, stddev, weights, missing_values,
                    data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                    data['SpacecraftAltitude'],
                    gamma[data['ColumnIndices']],
                    rho_est
                )
                
                #print '3'
                gamma = omi.compute_smoothing_parameter(40.0, 40.0)
                rho_est = 4
                wgrid = omi.psm_grid(wgrid,
                    data['Longitude'], data['Latitude'],
                    data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                    new_weight, errors,new_weight*0.9, weights, missing_values,
                    data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                    data['SpacecraftAltitude'],
                    gamma[data['ColumnIndices']],
                    rho_est
                )
                # The 90% of new_weight = std. dev. is a best guess comparing uncertainty
                # over land and sea
                #print '4'
                
                grid.norm() # divides by the weights (at this point, the values in the grid are multiplied by the weights)
                # Replace by the new weights later
                #wgrid.norm() # if you normalize wgrid the data is not as smooth as it could be
                wgrid.values = np.nan_to_num(np.array(wgrid.values))
                grid.values = np.nan_to_num(np.array(grid.values))
                
                
                
                #grid.values = grid.values/grid.weights
                #wgrid.values = wgrid.values/wgrid.weights
                
                
                #print 'counter = ', counter, ':', np.max(gridcoll.values), np.max(grid.values), np.max(wgrid.values)
                gridcoll.values += np.nan_to_num(grid.values)*np.nan_to_num(wgrid.values)
                gridcoll.weights += wgrid.values
                grid.zero()
                wgrid.zero()
                
                
                


            # 9) The distribution of values and errors has to be normalised
            #    with the weight.
            gridcoll.norm()
            #grid.norm()

            # 10) The Level 3 product can be saved as HDF5 file
            #     or converted to an image (requires matplotlib and basemap
            
            rho_est = 4e16
            gridcoll.save_as_he5('%s.he5' % (filename))
            #gridcoll.save_as_image('%s.png' % (filename), vmin=0, vmax=rho_est)

        except:
            print('No datas available at following day:', start_date)


    
    
    #grid.save_as_he5('%s_%s_%s_%s_%s.he5' % (grid_name, str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4]))
    #grid.save_as_image('%s_%s_%s_%s_%s.png' % (grid_name, str(start_date)[8:10],str(start_date)[5:7],  str(start_date)[0:4]), vmin=0, vmax=rho_est)
    #grid.save_as_image('%s_%s_%s_%s_%s.he5' % ( str(start_date)[8:10],  str(start_date)[5:7],  str(start_date)[0:4], grid_name, gridding_method), vmin=0, vmax=rho_est)

    # 11) It is possible to set values, errors and weights to zero.
    grid.zero()









if __name__ == '__main__':



    start = datetime.datetime(2015, 6, 1)
    stop = datetime.datetime(2015, 6, 2)
    
    grid_name = 'NorthAmericaBEHR'
    
    data_path = '/Volumes/share-sat/SAT/BEHR/WEBSITE/webData/PSM-Comparison'         # where OMI-raw- datas are saved
    #save_path = '/project/meteo/wenig/OMI_new_weight/Germany' # path, where you want to save your datas
    #save_path = '/project/meteo/wenig/OMI_new_weight/Germany_try' # path, where you want to save your datas
    save_path = '/Users/Josh/Documents/MATLAB/BEHR/Workspaces/PSM-Comparison/Tests/UpdateBranch' # path, where you want to save your datas
    
    
    #year = 2005
    #month = 1
    #day = 1
    #start_date = datetime.datetime(year,month,day)
    #end_date = datetime.datetime(year,month,day+1)
    #main(start_date, end_date, grid_name, data_path, save_path)
    
    #p = Pool(1)
    
    
    #a,b,c,d, e = generate_zip_lists(start, stop, grid_name, data_path, save_path)
    #pdb.set_trace()
    main(start, stop, grid_name, data_path, save_path)
    
    #p.map(main, list(zip(a,b,c,d,e)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  






