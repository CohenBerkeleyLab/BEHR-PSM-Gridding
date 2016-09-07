# /usr/bin/env python
# coding: utf-8
#HDF5_DISABLE_VERSION_CHECK=2

from datetime import datetime
from datetime import timedelta

import numpy as np
import numpy.ma as ma

import omi
import time
import platform
import h5py
import he5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import pylab as plt
import sys, getopt

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
    mask |= XTrackQualityFlags

    # set invalid cloud cover to 100% -> smallest weight
    CloudRadianceFraction[CloudRadianceFraction.mask] = 1.0

    # values and errors
    values = ma.array(ColumnAmountNO2Trop, mask=mask)
    errors = ma.array(ColumnAmountNO2TropStd, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al., 2008)
    #stddev = 1.5e15 * (1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    stddev = 1.5e15 *(1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    area = FoV75Area.reshape(1, FoV75Area.size)
    area = area.repeat(ColumnAmountNO2Trop.shape[0], axis=0)

    if gridding_method.startswith('psm'):
        #weights = ma.array(1.0 / area, mask=mask)
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)
    else:
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)
    #weights[weights < 0] = 0
    return values, errors, stddev, weights




def griddata(startdate, enddate, gridding_method, grid_name, data_path,grid, dispmessages):

    # 1. Define a grid
    # (a) by giving lower-left and upper-right corner
#    grid = omi.Grid(
 #       llcrnrlat=19.6, urcrnrlat=25.6,
  #      llcrnrlon=108.8, urcrnrlon=117.6, resolution=0.01
   # )
    #grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.1)#global
    #grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.01)#Germany
    # (b) or by reading this data from a JSON file
    #    (the default file can be found in omi/data/gridds.json)
    #grid = omi.Grid.by_name(grid_name)

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

    # 4a) data in OMI files can be read by
    # >>> data = omi.he5.read_datasets(filename, name2dataset)

    # 4b) or by iterating over orbits from start to end date at the following
    #   location: 
    #       os.path.join(data_path, product, 'level2', year, doy, '*.he5')
    #
    #   (see omi.he5 module for details)
    products = ['OMNO2.003', 'OMPIXCOR.003']
    for timestamp, orbit, data in omi.he5.iter_orbits(
            startdate, enddate, products, name2datasets, data_path
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
        if dispmessages:
            print 'time: %s, orbit: %d' % (timestamp, orbit)
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
    #     or converted to an image (requires matplotlib and basemap)
    #filename='test_%s_2012_Feb3_'% gridding_method
    #filename='NO2_%s'%gridding_method,

    if (enddate-startdate==timedelta(1)):
        filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s)'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    else:
        filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s-%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s)'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day, (enddate+timedelta(days=-1)).year,(enddate+timedelta(days=-1)).month,(enddate+timedelta(days=-1)).day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)

    if dispmessages:
        print 'writing: ',filename
    grid.save_as_he5(filename+'.he5')
    #grid.save_as_image(filename+'.png', vmin=0, vmax=rho_est)

    # 11) It is possible to set values, errors and weights to zero.
    grid.zero()

def test():
    grid = omi.Grid(
        llcrnrlat=-70.0, urcrnrlat=80.0,
        llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.1
    )
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
    #name2dataset = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]
    name2dataset = NAME2DATASET_NO2
    #print name2dataset
    # 4a) data in OMI files can be read by
    #data = omi.he5.read_datasets('I:\OMI\OMNO2.003\level3\Germany\NO2_psm_(2012_1_2)_lat(40.0_55.0)_lon(-5.0_20.0)_res(0.01).he5', NAME2DATASET_NO2)
##    data = [
##            ('lon', self.lon),
##            ('lat', self.lat),
##            ('values', self.values),
##            ('errors', self.errors),
##            ('weights', self.weights)
##        ]
## 
    #filename='I:\OMI\OMNO2.003\level3\Germany\NO2_psm_(2012_1_2)_lat(40.0_55.0)_lon(-5.0_20.0)_res(0.01).he5'
    #filename='I:\OMI\OMNO2.003\level3\global\NO2_psm_(2011_1_1)_lat(-70.0_80.0)_lon(-180.0_180.0)_res(0.1).he5'
    #filename='F:\programs\Python\Gridding\omigrid\examples\NO2_psm_(2005_11_20)_lat(-90.0_90.0)_lon(-180.0_180.0)_res(0.1)1.he5'
    filename='I:\OMI\OMNO2.003\level3\global\NO2_psm_(2005_1_1)_lat(-90.0_90.0)_lon(-180.0_180.0)_res(0.1).he5'
    with h5py.File(filename, 'r') as f:
        for name in f:
            print name
        errdata=f.get('errors');
        vdata=f.get('values');
        wdata=f.get('weights')
        #wdata=wdata*1e35;
        print type(wdata.value)
        #wdata= np.array(wdata)*1e35
        wdata=wdata.value*1e35
        print type(wdata)
        #print errdata.value[1::100,1::100]
        #print vdata.value[1::100,1::100]
        #print wdata[1::100,1::100]
        s=errdata.shape
        print s[0],s[1]

        
        plt.imshow(np.rot90(vdata),cmap='jet')
        #plt.clim(0.0,100)
        plt.show()
        d=wdata[1::100,1::100]
        #wdata[wdata<=0]=0
        np.set_printoptions(precision =3,suppress=False,threshold='nan',linewidth=160)
        print d
        d[np.isnan(d)]=0
        d[d<0]=0
        print np.int_(d)
        #print(np.vectorize("%.2f".__mod__)(d))


def gridname2grid(grid_name):
    if grid_name=='global':
        grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.05);# grid_name = 'global'#7200*3600
    elif grid_name=='Germany':
        grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.002);# grid_name = 'Germany'#7500*12500
    elif grid_name=='Europe':
        grid = omi.Grid(llcrnrlat=35.0, urcrnrlat=60.0,llcrnrlon=-10.0, urcrnrlon=25.0, resolution=0.005);# grid_name = 'Europe'#7000*7000
    elif grid_name=='Northamerica':
        grid = omi.Grid(llcrnrlat=15.0, urcrnrlat=55.0,llcrnrlon=-125.0, urcrnrlon=-65.0, resolution=0.01); #grid_name = 'Northamerica'#6000*4000
    elif grid_name=='Asia':
        grid = omi.Grid(llcrnrlat=20.0, urcrnrlat=43.0,llcrnrlon=100.0, urcrnrlon=142.0, resolution=0.005); #grid_name = 'Asia'#4200*2300
    return grid

def main(argv):
    startdate='not set'
    enddate='not set'
    area='global'
    dispmessages=False
    try:
        opts, args = getopt.getopt(argv,"hs:e:a:o:",["startdate=","enddate=","area=","output="])
    except getopt.GetoptError:
        print 'specify startdate (startdate=dd.mm.yyy), enddate (enddate=dd.mm.yyy), and aerea (area=global/Germany/Europe/Northamerica/Asia)'
        sys.exit(2)
    print opts
    print args
    for opt, arg in opts:
        if opt == '-h':
            print 'specify startdate (startdate=dd.mm.yyy), enddate (enddate=dd.mm.yyy), and aerea (area=global/Germany/Europe/Northamerica/Asia)'
            sys.exit()
        elif opt in ("-s", "--startdate"):
            startdate = arg
        elif opt in ("-e", "--enddate"):
            enddate = arg
        elif opt in ("-a", "--area"):
            area = arg
        elif opt in ("-o", "--output"):
            if arg not in ('no output','nooutput','no_output','no','false','False'):
                dispmessages=True
            #else:
    print 'startdate is ', startdate
    print 'enddate is ', enddate
    print 'area is ', area
    print 'displaymessages is ', dispmessages


    date_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
    #startdate=datetime.strptime(startdate, "%d.%m.%Y")#.date()
    #print startdate
    # "data_path" is the root path to your OMI data. Please change it for
    # you settings. 
    #
    # The OMI files are assumed to be location at:
    #    "{data_path}/{product}/level2/{year}/{doy}/*.he5"
    #
    # For example:
    #    "/home/gerrit/Data/OMI/OMNO2.003/level2/2006/123/*.he5"
    #

    if platform.system() == 'Windows':
        data_path = 'I:\OMI'
    else:         
        data_path = '/home/zoidberg/OMI'
    if dispmessages:
        print 'OMI data path: ',data_path
    
    # The start and end date of Level 2 data you want to grid. The enddate
    # is NOT included!
    #startdate = datetime(2011,7,16)
    #enddate = datetime(2012,1,1)
    startdate = datetime.strptime(startdate, "%d.%m.%Y")#.date()#datetime(2005,7,7)
    enddate = datetime.strptime(enddate, "%d.%m.%Y")#.date()#datetime(2011,1,1)

    # Name of the level 3 grid
    #grid_name = 'prd'
    #grid_name = 'Germany'
    #grid_name = 'global'
    grid_name=area;
    if grid_name=='global':
        grid = omi.Grid(llcrnrlat=-90.0, urcrnrlat=90.0,llcrnrlon=-180.0, urcrnrlon=180.0, resolution=0.05);# grid_name = 'global'#7200*3600
    elif grid_name=='Germany':
        grid = omi.Grid(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, resolution=0.002);# grid_name = 'Germany'#7500*12500
    elif grid_name=='Europe':
        grid = omi.Grid(llcrnrlat=35.0, urcrnrlat=60.0,llcrnrlon=-10.0, urcrnrlon=25.0, resolution=0.005);# grid_name = 'Europe'#7000*7000
    elif grid_name=='Northamerica':
        grid = omi.Grid(llcrnrlat=15.0, urcrnrlat=55.0,llcrnrlon=-125.0, urcrnrlon=-65.0, resolution=0.01); #grid_name = 'Northamerica'#6000*4000
    elif grid_name=='Asia':
        grid = omi.Grid(llcrnrlat=20.0, urcrnrlat=43.0,llcrnrlon=100.0, urcrnrlon=142.0, resolution=0.005); #grid_name = 'Asia'#4200*2300
    #grid_name = 'test'

    # Call main function twice to grid data using CVM and PSM
    #main(startdate, enddate, 'psm', grid_name, data_path)
    starttime=time.clock()


    diff = enddate - startdate
    for d in range(diff.days + 1):
        if dispmessages:
            print startdate+timedelta(d)
        griddata(startdate+timedelta(d),startdate+timedelta(d+1), 'psm', grid_name, data_path, grid, dispmessages)
        #griddata(startdate+timedelta(d),startdate+timedelta(d+1), 'cvm', grid_name, data_path, grid, dispmessages)
    #griddata(startdate, enddate, 'psm', grid_name, data_path)
    #test(startdate, enddate,'psm', grid_name, data_path)

    #test()
    #griddata(startdate,startdate+timedelta(1), 'psm', grid_name, data_path, grid, dispmessages)
    endtime=time.clock()
    if dispmessages:
        print 'time: ',(endtime-starttime)/60,' min'

def show_map(vdata,errdata,wdata,zoom,date):
    if 'fig' not in globals():            
        global fig, ax1, ax2, ax3
        fig=plt.figure(figsize=(10,15))
        ax1=fig.add_subplot(3,1,1)
        ax2=fig.add_subplot(3,1,2)
        ax3=fig.add_subplot(3,1,3)
        #fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    #fig.set_figheight(30)
    #fig.set_figwidth(20)
    #plt.figure(num=1, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')
    #fig.set_size_inches(1,1)
        
    #fig.patch.set_facecolor('white')
    #rect.set_facecolor('white')
    fig.suptitle('OMI data '+date.strftime("%d.%m.%Y"), fontsize=20)
    ax1.set_title('NO$_2$ SCDs [$10^{16}$ molec./cm$^2$]')
    if zoom:
        im1 = ax1.imshow(vdata[3600-3000:3600-2000,3500:4500]/1e16, vmin=0, vmax=1, cmap='jet', aspect='auto')
    else:
        im1 = ax1.imshow(vdata/1e16, vmin=0, vmax=1, cmap='jet', aspect='auto')
    ax1.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    #cax1.yaxis.set_visible(False)
    cax1.xaxis.set_visible(False)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax2.set_title('NO$_2$ SCD errors [$10^{16}$ molec./cm$^2$]')
    if zoom:
        im2 = ax2.imshow(errdata[3600-3000:3600-2000,3500:4500]/1e16, vmin=0, vmax=1, cmap='jet', aspect='auto')
    else:
        im2 = ax2.imshow(errdata/1e16, vmin=0, vmax=1, cmap='jet', aspect='auto')
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    #cax2.yaxis.set_visible(False)
    cax2.xaxis.set_visible(False)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax3.set_title('weights')
    if zoom:
        im3 = ax3.imshow(wdata[3600-3000:3600-2000,3500:4500], vmin=20, vmax=120, cmap='jet', aspect='auto')
    else:
        im3 = ax3.imshow(wdata, vmin=20, vmax=120, cmap='jet', aspect='auto')
    ax3.yaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    #cax3.yaxis.set_visible(False)
    cax3.xaxis.set_visible(False)
    cbar3 = plt.colorbar(im3,cax=cax3)
          
    #plt.show(block=False)
    plt.draw()
    plt.pause(0.01)
    #d=wdata[1::100,1::100]
    d=errdata[3500:4500:100,2000:3000:100]
    #wdata[wdata<=0]=0
    np.set_printoptions(precision =3,suppress=False,threshold='nan',linewidth=160)
    #print d


def generate_filename(startdate,enddate,grid_name,gridding_method,suffix):
    if platform.system() == 'Windows':
        data_path = 'I:\OMI'
    else:         
        data_path = '/home/zoidberg/OMI'
    grid=gridname2grid(grid_name)
    #filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).he5'% (data_path,grid_name,gridding_method,mapdate.year,mapdate.month,mapdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    if (enddate-startdate==timedelta(1)):
        filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).%s'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution,suffix)
    else:
        filename='%s/OMNO2.003/level3/av_%s/NO2_%s_(%s_%s_%s-%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).%s'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day,enddate.year,enddate.month,enddate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution,suffix)
    return filename

def read_map(mapdate,grid_name,gridding_method):
    NAME2DATASET_PIXEL = {}
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields',['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude','FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],NAME2DATASET_PIXEL)
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields',['Latitude', 'Longitude', 'SpacecraftAltitude','SpacecraftLatitude', 'SpacecraftLongitude'],NAME2DATASET_PIXEL)

    NAME2DATASET_NO2 = {}
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields',['CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2Trop','ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit','VcdQualityFlags', 'XTrackQualityFlags'],NAME2DATASET_NO2)
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields',['SolarZenithAngle', 'Time'],NAME2DATASET_NO2)
    #name2dataset = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]
    name2dataset = NAME2DATASET_NO2
    #filename='I:\OMI\OMNO2.003\level3\global\NO2_psm_(2005_1_1)_lat(-90.0_90.0)_lon(-180.0_180.0)_res(0.05).he5'
    if platform.system() == 'Windows':
        data_path = 'I:\OMI'
    else:         
        data_path = '/home/zoidberg/OMI'
    grid=gridname2grid(grid_name)
    filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).he5'% (data_path,grid_name,gridding_method,mapdate.year,mapdate.month,mapdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)

    print filename
    
    with h5py.File(filename, 'r') as f:
        #for name in f:
        #    print name
        errdata=np.rot90(np.copy(f.get('errors').value))
        vdata=np.rot90(np.copy(f.get('values').value))
        wdata=np.rot90(np.copy(f.get('weights').value*1e35))
        lat=f.get('lat').value
        lon=f.get('lon').value
    return vdata,errdata,wdata,lon, lat

def read_maps(startdate,enddate,grid_name,gridding_method):
    NAME2DATASET_PIXEL = {}
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields',['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude','FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],NAME2DATASET_PIXEL)
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields',['Latitude', 'Longitude', 'SpacecraftAltitude','SpacecraftLatitude', 'SpacecraftLongitude'],NAME2DATASET_PIXEL)

    NAME2DATASET_NO2 = {}
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields',['CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2Trop','ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit','VcdQualityFlags', 'XTrackQualityFlags'],NAME2DATASET_NO2)
    omi.he5.create_name2dataset('/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields',['SolarZenithAngle', 'Time'],NAME2DATASET_NO2)
    #name2dataset = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]
    name2dataset = NAME2DATASET_NO2
    #filename='I:\OMI\OMNO2.003\level3\global\NO2_psm_(2005_1_1)_lat(-90.0_90.0)_lon(-180.0_180.0)_res(0.05).he5'
##    if platform.system() == 'Windows':
##        data_path = 'I:\OMI'
##    else:         
##        data_path = '/home/zoidberg/OMI'
##    grid=gridname2grid(grid_name)
    #filename='%s/OMNO2.003/level3/%s/NO2_%s_(%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).he5'% (data_path,grid_name,gridding_method,mapdate.year,mapdate.month,mapdate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    filename=generate_filename(startdate,enddate,grid_name,gridding_method,'he5')
    print 'reading ',filename
    
    with h5py.File(filename, 'r') as f:
        #for name in f:
        #    print name
        errdata=np.rot90(np.copy(f.get('errors').value))
        vdata=np.rot90(np.copy(f.get('values').value))
        wdata=np.rot90(np.copy(f.get('weights').value*1e35))
        lat=f.get('lat').value
        lon=f.get('lon').value
    return vdata,errdata,wdata,lon, lat



def average_days(startdate,enddate,grid_name,gridding_method,displaymaps):
    if displaymaps:
        plt.ion()
    grid=gridname2grid(grid_name)
    s1=(grid.urcrnrlat-grid.llcrnrlat)/grid.resolution
    s2=(grid.urcrnrlon-grid.llcrnrlon)/grid.resolution

    avmap=np.zeros((s1,s2))
    avweight=np.zeros((s1,s2))
    averr=np.zeros((s1,s2))
    #avweight=np.zeros((s1,s2))
    diff = enddate - startdate
    for d in range(diff.days + 1):
        print startdate+timedelta(d)
    
        try:
            vdata,errdata,wdata,lon,lat=read_map(startdate+timedelta(d),grid_name,gridding_method)
        except IOError, e:
            print 'Error reading file: '
            print e
        else:
            #print 'wdata',wdata[1,1], wdata[1010:1015,110:115]
            #print 'vdata',vdata[1,1],vdata[1010:1015,110:115]
            wdata[np.isnan(wdata)]=0
            vdata[np.logical_and(np.isnan(vdata),~np.isnan(avmap))]=0
            wdata[wdata>100]=100
            wdata[np.logical_and(wdata<50,wdata>0)]=50
            #wdata[wdata<50]=50
            wdata[wdata<0]=0
            avmap+=vdata*wdata
            avweight+=wdata
            averr+=wdata**2+errdata**2
            if displaymaps:
                print 'wdata_',wdata[1,1],wdata[1010:1015,110:115]
                print 'avmap',avmap[1,1],avmap[1010:1015,110:115]
                print 'avweight',avweight[1,1],avweight[1010:1015,110:115]
                #show_map(vdata,errdata,wdata,False,startdate+timedelta(d))
                showmap=np.divide(avmap,avweight)
                showmap[showmap==0]=np.nan
                show_map(vdata,showmap,wdata,False,startdate+timedelta(d))

                #d=wdata[1::100,1::100]
                d=wdata[3600-3000:3600-2000:10,3500:4500:10]
                #wdata[wdata<=0]=0
                np.set_printoptions(precision =3,suppress=False,threshold='nan',linewidth=160)
                #print d
    #print vdata.shape
    data = [
            ('lon', lon),
            ('lat', lat),
            ('values', avmap/avweight),
            ('errors', np.sqrt(averr)),
            ('weights', avweight)
        ]
    if platform.system() == 'Windows':
        data_path = 'I:\OMI'
    else:         
        data_path = '/home/zoidberg/OMI'

    filename='%s/OMNO2.003/level3/av_%s/NO2_%s_(%s_%s_%s-%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).he5'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day, enddate.year,enddate.month,enddate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    print 'writing ', filename
    he5.write_datasets(filename, data)
    if displaymaps:
        plt.ioff()
        plt.show()

def convert_map2jpg(startdate,enddate,grid_name,gridding_method,vmin,vmax):
    vdata,errdata,wdata,lon,lat=read_maps(startdate,enddate,grid_name,gridding_method)
    data = [('lon', lon),('lat', lat),('values', vdata),('errors', errdata),('weights', wdata)]
    if platform.system() == 'Windows':
        data_path = 'I:\OMI'
    else:         
        data_path = '/home/zoidberg/OMI'
    grid=gridname2grid(grid_name)
    #filename='%s/OMNO2.003/level3/av%s/NO2_%s_(%s_%s_%s-%s_%s_%s)_lat(%s_%s)_lon(%s_%s)_res(%s).jpg'% (data_path,grid_name,gridding_method, startdate.year,startdate.month,startdate.day, enddate.year,enddate.month,enddate.day, grid.llcrnrlat, grid.urcrnrlat, grid.llcrnrlon, grid.urcrnrlon,grid.resolution)
    filename=generate_filename(startdate,enddate,grid_name,gridding_method,'jpg')
    print 'writing ', filename
    #he5.write_datasets(filename, data)


    #import matplotlib.pyplot as plt
##    from mpl_toolkits.basemap import Basemap
##
##    fig = plt.figure(figsize=(8, 8*self.ratio))
##    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
##
##    m = Basemap(ax=ax, resolution='i', **self.to_basemap())
##    m.drawcoastlines()
##    res = m.imshow(data.values.T, vmin=vmin, vmax=vmax)
##
##    fig.savefig(filename, dpi=500)
##    plt.close(fig)
    s=vdata.shape
    mydpi=100
    plt.figure(figsize=(s[0]/mydpi,s[1]/mydpi), dpi=mydpi)
    vdata[np.isnan(vdata)]=0
    vdata[vdata<vmin]=vmin
     
    fig=plt.imshow(np.rot90(np.log(vdata),3), vmin=np.log(vmin), vmax=np.log(vmax), cmap='jet', aspect='auto')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    #plt.show()
    
    plt.savefig(filename,dpi=mydpi, bbox_inches='tight', pad_inches=0)

    
if __name__ == '__main__':
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    mapname='Germany'
    #average_days(datetime(2006,1,1),datetime(2006,1,3),mapname,'psm',False)
    average_days(datetime(2005,1,1),datetime(2005,2,1),mapname,'psm',False)
##    average_days(datetime(2006,1,1),datetime(2006,12,31),mapname,'psm',False)
##    average_days(datetime(2007,1,1),datetime(2007,12,31),mapname,'psm',False)
##    average_days(datetime(2008,1,1),datetime(2008,12,31),mapname,'psm',False)
##    average_days(datetime(2009,1,1),datetime(2009,12,31),mapname,'psm',False)
##    average_days(datetime(2010,1,1),datetime(2010,12,31),mapname,'psm',False)
##    average_days(datetime(2011,1,1),datetime(2011,12,31),mapname,'psm',False)
##    average_days(datetime(2012,1,1),datetime(2012,12,31),mapname,'psm',False)
##    #average_days(datetime(2013,1,1),datetime(2013,12,31),mapname,'psm',False)
    #average_days(datetime(2005,1,1),datetime(2012,12,31),mapname,'psm',False)


    
    #convert_map2jpg(datetime(2005,1,1),datetime(2012,12,31),'global','psm',1e14,1.5e16)
    #convert_map2jpg(datetime(2005,1,1),datetime(2005,12,31),'Germany','psm',1e14,2e16)
    #convert_map2jpg(datetime(2005,1,1),datetime(2012,12,31),'Germany','psm',1e14,2e16)
    #main(sys.argv[1:])

    #average_days(datetime(2005,1,1),datetime(2013,12,31),'global','cvm',False)
    #convert_map2jpg(datetime(2005,1,1),datetime(2013,12,31),'global','cvm',1e14,2e16)
    convert_map2jpg(datetime(2005,1,1),datetime(2005,1,1),'global','cvm',1e14,2e16)
