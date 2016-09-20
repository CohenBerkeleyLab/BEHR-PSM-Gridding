import glob
from datetime import date
import calendar
import h5py
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import re

#--------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


# EuropeList
AmericaList =[['San Francisco', -122.419416,37.774929],
             ['Los Angeles', -118.243685, 34.052234], 
             ['New York City', -74.005941, 40.712784],
             ['Chicago', -87.629798, 41.878114],
             ['Dallas', -96.796988, 32.776664],
             ['Detroit', -83.045754, 42.331427],
             ['Houston', -95.369803, 29.760427],
             ['Mexico City', -99.133208, 19.432608],
             ['Brasilia', -47.882166, -15.794229],
             ['Salvador', -38.502304,-12.97304],
             ['Fortaleza', -38.52667, -3.731862],
             ['Buenes Aires', -58.381559, -34.603684],
             ['Santiago de Chile', -70.669265, -33.44889],
             ['La Paz', -68.119294, -16.489689],
             ['Lima', -77.042793, -12.046374],
             ['Bogota', -74.072092, 4.710989]]


List = AmericaList
#def sort(st)



def cities(m, List):        
    for i in range(len(List)):
        m.scatter(x = List[i][1], y=List[i][2], c = "red", latlon = True, alpha = 0.8)
        x, y = m(List[i][1], List[i][2])
        plt.annotate(List[i][0],xy =(x, y), xytext=(x+0.1, y+0.1), color='red')#, arrowprops=dict(arrowstyle="->") )#
    return m    




 


def readin(Filename):
    f = h5py.File(Filename,'r')
    dictionary = {}
    for name in f:
         str = 'array'
         str2 = str+'_%s' %np.str(name) # = np.get(name)
         #print str2
         dictionary[str2] =  f.get(name)
    return  np.array(dictionary['array_values']), np.array(dictionary['array_errors']), np.array(dictionary['array_weights']), np.array(dictionary['array_lat']), np.array(dictionary['array_lon'])
                                                                                                                                                                      
                                                                                                                                                                      



def average(collectinglist):
    
    
    values, errors, weights, lat, lon = readin(collectinglist[0]) 
    del errors, lat, lon 
    valuesbox = np.zeros((np.shape(values)[0], np.shape(values)[1]))
    weightsbox = np.zeros((np.shape(values)[0], np.shape(values)[1]))
    #print np.shape(values)
    counter = 0
    #print np.type(valuesbox), np.type(valuesmap)
    
    for filename in collectinglist:
        #print filename
        print counter +1, '/', len(collectinglist), collectinglist[counter]
        values, errors, weights, lat, lon = readin(filename)
        weights = np.nan_to_num(weights)
        values = np.nan_to_num(values)
        valuesbox += values*weights
        #print 'shape weightsbox', np.shape(weightsbox)
        weightsbox += weights
        #print 'shape weightsbox', np.shape(weightsbox)
        counter = counter + 1
        #valuemap.append(weights)
    
    valuesbox = valuesbox/weightsbox

    return valuesbox.T[::-1]   #Transponieren und dann noch Matrix auf den Kopf drehen



def plotaverage(a, plottitle, region, List):
    #m = Basemap(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Deutschland
    #m = Basemap(llcrnrlat=40.0, urcrnrlat=55.0,llcrnrlon=-5.0, urcrnrlon=20.0,resolution='i') #Deutschland
    #m = Basemap(llcrnrlat=15.0, urcrnrlat=55.0,llcrnrlon=-125.0, urcrnrlon= -65.0,resolution='i') #North America
    #m = Basemap(llcrnrlon = 35.5,llcrnrlat=32.05,urcrnrlon = 42.5,urcrnrlat = 37.55, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Syrien
    #m = Basemap(llcrnrlat=19.6, urcrnrlat=25.6,llcrnrlon=108.8, urcrnrlon=117.6, rsphere=(6378137.00,6356752.3142),resolution='i')#,projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Pearl River Delta
    #m = Basemap(llcrnrlat=45.7, urcrnrlat=55.5,llcrnrlon=4.5, urcrnrlon=17.5, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Deutschland
    #m = Basemap(llcrnrlat=-36.0, urcrnrlat=40.0,llcrnrlon=-19.0, urcrnrlon=52.0, rsphere=(6378137.00,6356752.3142),resolution='i')#,projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Afrika
    #m = Basemap(llcrnrlat=-11.0, urcrnrlat=55.0, llcrnrlon=70.0, urcrnrlon= 150.0, rsphere=(6378137.00,6356752.3142),resolution='i')#,projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Asien
    #m = Basemap(llcrnrlat=30.0, urcrnrlat=75.0,llcrnrlon=-30.0, urcrnrlon=50.0, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) #Europa
    #m = Basemap(llcrnrlat=-85.0, urcrnrlat=85.0,llcrnrlon=  -180.0, urcrnrlon= 180.0, resolution='c') #, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) # nfast alles
    
    m = Basemap(llcrnrlat= 25.0, urcrnrlat= 50.0,llcrnrlon=  -125.0, urcrnrlon= -65.0, resolution='i') #, rsphere=(6378137.00,6356752.3142),resolution='i',projection='merc', lat_0=40.,lon_0=-20.,lat_ts=20.) # nfast alles

    #[-11.0, 55.0,   70.0, 150.0, 0.02],
    #m.imshow(a, interpolation='nearest', origin='upper', cmap = 'Paired')
    #m.imshow(a, interpolation='nearest', origin='upper', cmap = 'spectral', vmax = 6.4e15)
    img = m.imshow(a, interpolation='nearest', origin='upper', cmap = 'terrain', vmin = 0, vmax = 1e16) #jet, terrain, gnuplot
    #m.imshow(a, interpolation='nearest', origin='upper', cmap = 'terrain', vmin = 0, vmax = 6.5e16) #jet, terrain, gnuplot
    m.drawcoastlines()
    m.drawcountries()
    m.drawcounties()
    m.colorbar(img, label = 'NO2 vertical column densitiy')

    m = cities(m,List)
    
    #plt.imshow(image, cmap = 'terrain')
    #plt.colorbar(label = "terrain")
  
        
        
    #plt.title('NO2-Konzentrationen Deutschland %s, %s' %(year, weekdays[0]))
    plt.title(plottitle)
    
    savetitel  = plottitle.replace(", ", "_")
    #lastsequenzstart = files[0].rfind('/')
    #name = files[0][:lastsequenzstart+1]
    plt.savefig('%s' %(savetitel))#', format="png")#, bbox_inches= 'tight', pad_inches=0.5)
    np.save('%s.npy' %(savetitel), a)   #load with """ b = np.load('a.npy')  """


    
    plt.show()
    #plt.close()
    

    
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------




year = 2015
region = "North America"
collectinglist = glob.glob('*america*.he5')
average_hole = average(collectinglist)
plottitle = "PSM, BEHR, w_g"
plotaverage(average_hole, plottitle, region, List)







