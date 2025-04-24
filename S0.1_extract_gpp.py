import ee

# Initialize the Earth Engine module
ee.Initialize()

# Define the area of interest (AOI)
aoi = ee.FeatureCollection("projects/ee-zhengfei/assets/pf_shape")

projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection()

# Define the start and end years
start_year = 2000
end_year = 2023
folder = 'gpp_pf'

roi = ee.Geometry.Rectangle([-180, 27.05, 180, 83.67],'EPSG:4326', False);


# Function to calculate yearly GPP and export each year's data
for year in range(start_year, end_year + 1):
    imagecollections = []
    for n in range(23):
        start = ee.Date(str(year)+'-01-01').advance(n*16, 'day')
        end = start.advance(16, 'day')
        gpp = ee.ImageCollection('MODIS/006/MOD17A2H') \
                  .filter(ee.Filter.date(start, end)) \
                  .select('Gpp').mean() \
        .clip(roi)#.setDefaultProjection(projection_modis) \
        #     .reduceResolution(ee.Reducer.mean(), False, 12000) \
        #     .reproject('EPSG:4326', None, 9278)
        imagecollections.append(gpp)

    #
    yearly_gpp = ee.ImageCollection.fromImages(imagecollections).toBands().clip(roi).setDefaultProjection(projection_modis) \
        .reduceResolution(ee.Reducer.mean(), False, 12000) \
        .reproject('EPSG:4326', None, 9278)

    # Set the export task
    task = ee.batch.Export.image.toDrive(
        image=yearly_gpp,
        folder=folder,
        description='Gpp_pf_' + str(year),#+'_'+str(n),
        scale=9278,
        region=roi
    )

    # Start the export task
    task.start()
    print(str(year))

# tasks = ee.batch.Task.list()
# tasks[0].status()
# #
# for task in ee.batch.Task.list():
# 	task.cancel()