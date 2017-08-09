# Author: Annette Schuett, Gerret Kuhlmann


# /usr/bin/env python
# coding: utf-8

import datetime

import numpy as np
import numpy.ma as ma
from multiprocessing import Pool
import os.path
from scipy import signal as sig

#import sys
#sys.path.append('/usr/users/annette.schuett/Masterarbeit/omi-master/omi')
import omi

#import '/usr/users/annette.schuett/Masterarbeit/omi-master/omi'

#      new programm, without mask |= RootMeanSquareErrorOfFit > 0.0003, because the new product has no RootMeanSquareErrorOfFit

import pdb


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
     'ColumnAmountNO2TropStd',
     'VcdQualityFlags', 'XTrackQualityFlags'],
    NAME2DATASET_NO2
)
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields',
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
    
    #print mask
    

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
    cf = CloudRadianceFraction#*1000
    
    print np.nanmax(cf)
    weights = ma.array(1.0 / area, mask=mask)
    weights = ma.array(weights/((errors/1e16)*(1.3 + 0.87*cf))**2, mask=mask)

    return values, errors, stddev, weights, cf


def generate_zip_lists(start, stop, grid_name, data_path, save_path, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid):
    startlist = []
    stoplist = []
    gridlist = []
    pathlist = []
    savelist = []
    x_con_num_list = []
    y_con_num_list = []
    x_con_mid_list = []
    y_con_mid_list = []
    
    
    
    
    daydate = start 
    while daydate != stop:
        startlist.append(daydate)
        daydate += datetime.timedelta(days = 1)
        stoplist.append(daydate)
        gridlist.append(grid_name)
        pathlist.append(data_path)
        savelist.append(save_path)
        x_con_num_list.append(x_convolution_number)
        y_con_num_list.append(y_convolution_number)
        x_con_mid_list.append(x_convolution_mid)
        y_con_mid_list.append(y_convolution_mid)
    
    return startlist, stoplist, gridlist, pathlist, savelist,x_con_num_list, y_con_num_list, x_con_mid_list, y_con_mid_list

def gridname2grid(grid_name): #return grid
    if grid_name=='Global':
        grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.05);# grid_name = 'Global' # 3600 x 7200
        #grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.01);# grid_name = 'Global' # 5*3600 x 5*7200
        
        
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
    elif grid_name=='Brasil_Acridicon':
        grid = omi.Grid(llcrnrlat= -19.0, urcrnrlat= 13.0,llcrnrlon= -84.0, urcrnrlon= -32.0, resolution=0.008); #grid_name = 'Brasilia' # 5500 * 5425
        #grid = omi.Grid(llcrnrlat= -19.0, urcrnrlat= 13.0,llcrnrlon= -84.0, urcrnrlon= -32.0, resolution=0.08); #grid_name = 'Brasilia' # 5500 * 5425 
        
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
        

    return grid


def generate_filename(save_path, startdate,grid_name):

    
    grid=gridname2grid(grid_name)
    filename='%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s)'% (save_path, grid_name, startdate.year, startdate.month, startdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    
    return filename



def Convolution(Big):
    Small = np.array([[1,90,1],[15,100,15],[1,90,1]])
    #Small = np.array([[1,1,1],[1,2,1],[1,1,1]])
    return sig.convolve2d(Big,Small, mode='full', boundary='fill', fillvalue=0)

#def Convolution(Big):
    ##Small = np.array([[1,1,1],[1,10,1],[1,1,1]])
    ##Small = np.array([[1,1,1,1,1,],[1,1,1,1,1],[1,1,10,1,1,],[1,1,1,1,1],[1,1,1,1,1]])
    #Small = np.array([[1,10,10,10,1,],[1,10,10,10,1],[1,10,100,10,1,],[1,10,10,10,1],[1,10,10,10,1]])
    #return sig.convolve2d(Big,Small, mode='full', boundary='fill', fillvalue=0)

def Convolution1dx(Big, x_middle):
        Small = np.array([[1,x_middle,1]])
        return sig.convolve2d(Big,Small, mode='full', boundary='fill', fillvalue=0)



def Convolution1dy(Big, y_middle):
        Small = np.array([[1,y_middle,1]]).T
        return sig.convolve2d(Big,Small, mode='full', boundary='fill', fillvalue=0)


def redo_Convolution(Big, mx, my, x_middle, y_middle):
    pdb.set_trace()
    a = np.mean(Big)
    #print np.shape(Big)
    for i in range(mx):
        #print mx,i
        Big =  Convolution1dx(Big, x_middle)
        #Big = Big[0:len(Big),1:len(Big[0])-1]
        #print np.shape(Big)
    
    for j in range(my):
        #print my,j
        Big = Convolution1dy(Big, y_middle)
        #Big = Big[1:len(Big)-1,0:len(Big[0])]
        #print np.shape(Big)

    pdb.set_trace()
    Big = Big[my:len(Big)-my,mx:len(Big[0])-mx]
    return Big*(a/np.mean(Big))


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
def main(start_date, end_date, grid_name, data_path, save_path, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid):
    
    # 1. Define a grid
    
    grid = gridname2grid(grid_name)
    gridcoll = gridname2grid(grid_name)
    wgrid = gridname2grid(grid_name)
    
    grid.values = grid.values * 0.0
    wgrid.values = wgrid.values * 0.0
    gridcoll.values = gridcoll.values * 0.0
    
    
    
    filename  = generate_filename(save_path, start_date,grid_name)
    
    
    #fname = '%s.he5' % (filename)
    fname = '%s.he5' % (filename)
    if os.path.isfile(fname) == True:
        print 'Existing file:         ',  fname
    
    
    else:
        #print x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid
        
        try:
            print filename
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
            
            products = ['OMNO2.003', 'OMPIXCOR.003']
            for timestamp, orbit, data in omi.he5.iter_orbits(
                    start_date, end_date, products, name2datasets, data_path
                ):
                print 'time: %s, orbit: %d' % (timestamp, orbit)
                grid = gridname2grid(grid_name)
                wgrid = gridname2grid(grid_name)
                
                
                
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
                
                
                
                
                values, errors, stddev, weights, cf = preprocessing(**data)
                missing_values = values.mask.copy()
                
                #np.save('values.npy',values)
                values.dump('values.npy')
                weights.dump('weights.npy')

                # JLL 9 Aug 2017: for the Gaussian smoothing, it's necessary that all values being smoothed are valid.
                # From Annette: use a value that is typical of the area in question. So I will probably look at the
                # background average for the US to find an appropriate value here.
                #
                # This matters because the unmasked data is passed to the convolution, redo_Convolution(A.data, ...)
                # Afterwards the previously masked values are remasked, so these values should only be used in the
                # Gaussian smoothing.
                values.data[values.data<-1e29]=1e15
                values.data[values.data==np.nan] = 1e15
                
                valuesmask =  values.mask
                
                
                meanvalue = np.nanmean(values)
                
                #print 'mask', np.shape(valuesmask)
                if np.all(values.mask):
                    continue
                
                
                b = np.where(values >4*np.std(values))
                try:
                    # JLL 08 Aug 2017: For each value that is above the threshold, find the 3x3 grid of values around it
                    #
                    for i in range(len(b[0])):
                        #print i
                        #print b[0][i], b[1][i], values[b[0][i]][b[1][i]]
                        m = b[0][i]
                        n = b[1][i]
                        B = values[m-1:m+2,n-1:n+2]
                        #print B ,m,n
                        B0 = B*1.0 # JLL 08 Aug 2017: I'm assuming this is to make B0 independent of B?
                        B0[1][1] = np.nan
                        #print B[1][1]/np.nanmean(B[0])
                        if B[1][1]/np.nanmean(B[0])>= 30:
                            pdb.set_trace()
                            A = values[m-8:m+9,n-1:n+2]
                            amean = np.nanmean(A)
                            replace = redo_Convolution(A.data, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid)
                            replace2 = replace*(amean/np.nanmean(replace))
                            values[m-8:m+9,n-1:n+2] = replace2
                except:
                    print 'no std to high'
                
                values = ma.array(values, mask = valuesmask)
                new_weight = weights


                #mask0 = valuesmask
                #mask0 |= values.data ==np.nan
                #values = ma.array(values, mask = valuesmask*mask0)
                #meanconvalues = np.nanmean(values)
                #values = ma.array(values, mask = valuesmask*mask0)*(meanvalue/meanconvalues)
                #print np.shape(values), type(values)
                #values.dump('values3.npy')
                
                
                #print 'mean', np.nanmean(values), meanvalue
                
                
                #print np.max(weights), np.min(weights), np.max(values), np.min(values)
                
                print 'time: %s, orbit: %d' % (timestamp, orbit)
                
                gamma = omi.compute_smoothing_parameter(40.0, 40.0)
                #rho_est = 0.01
                rho_est = np.max(new_weight)*1.2
                wgrid = omi.psm_grid(wgrid,
                    data['Longitude'], data['Latitude'],
                    data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                    new_weight, errors,new_weight*0.9, weights, missing_values,
                    data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                    data['SpacecraftAltitude'],
                    gamma[data['ColumnIndices']],
                    rho_est
                )
                
                print 'wgrid vorbei'
                
                #rho_est = 4e16
                rho_est = np.max(values)*1.2
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
                
                print 'grid vorbei'
                grid.norm()
                wgrid.norm()
                
                
                
                wgrid.values = np.clip(np.array(wgrid.values), 0.01, np.max(np.array(wgrid.values)))
                
                pdb.set_trace()
                mask = ~np.ma.masked_invalid(grid.values).mask
                gridcoll.values += np.nan_to_num(grid.values)*np.nan_to_num(wgrid.values)*mask
                gridcoll.weights += np.nan_to_num(wgrid.values)*mask
                
                grid.zero()
                wgrid.zero()
                print 'gridzero, wgridzero'
                
                        
                        
            


            # 9) The distribution of values and errors has to be normalised
            #    with the weight.
            print 'doof'
            gridcoll.norm()

            print 'doof2'

            print filename
            gridcoll.save_as_he5('%s_clip_end.he5' % (filename))
            #gridcoll.save_as_image('%s.png' % (filename), vmin=0, vmax=rho_est)
            print 'geschafft'
            gridcoll.zero()

        except:
            print 'No datas available at following day:', start_date






if __name__ == '__main__':



    start = datetime.datetime(2013, 6, 1)    #Y/M/D
    #stop = datetime.datetime(2015, 11, 1)
    stop = datetime.datetime(2013,6, 2)
    
    
    
    #start = datetime.datetime(2005, 8, 1)    #Y/M/D
    ##stop = datetime.datetime(2015, 11, 1)
    #stop = datetime.datetime(2005, 8, 2)
    
    grid_name = 'Northamerica'
    #grid_name = 'Austria'
    #grid_name = 'Global'
    #grid_name = 'Germany'
    #grid_name = 'Brasil_Acridicon'
    #grid_name = 'Industrial_China'
    
    #data_path = '/home/zoidberg/OMI'         # where OMI-raw- datas are saved
    data_path = '/Volumes/share-sat/SAT/OMI/PSM_Links'         # where OMI-raw- datas are saved
    
    
    #save_path = '/project/meteo/wenig/OMI_new_weight/Germany' # path, where you want to save your datas
    #save_path = '/project/meteo/wenig/OMI_new_weight/Germany_try' # path, where you want to save your datas
    #save_path = '/home/a/Annette.Schuett/OMI/wichtig_gut/mapcalculation' # path, where you want to save your datas
    #save_path = "/project/meteo/wenig/Versuche/Germany"
    #save_path = "/project/meteo/wenig/OMI_new_weight/new"
    #save_path = "/project/meteo/wenig/OMI_new_weight/perfect_gauss"
    save_path = "."
    
    #year = 2005
    #month = 1
    #day = 1
    #start_date = datetime.datetime(year,month,day)
    #end_date = datetime.datetime(year,month,day+1)
    #main(start_date, end_date, grid_name, data_path, save_path)
    
    p = Pool(4)
    
    
    
    
    
    
    x_convolution_number= 3
    y_convolution_number= 8
    x_convolution_mid = 10
    y_convolution_mid = 2
    
    #main(start, stop, grid_name, data_path, save_path, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid)
    #a,b,c,d, e,f,g,h,i = generate_zip_lists(start, stop, grid_name, data_path, save_path, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid)
    
    
    #p.map(main, zip(a,b,c,d,e,f,g,h,i))
    main(start, stop, grid_name, data_path, save_path, x_convolution_number, y_convolution_number, x_convolution_mid, y_convolution_mid)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  






