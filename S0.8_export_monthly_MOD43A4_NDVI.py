import ee
from datetime import datetime, timedelta

# earthengine authenticate
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

shapefile = ee.FeatureCollection('projects/ee-zhengfei/assets/pf_shape')
dataset = ee.ImageCollection('MODIS/MCD43A4_006_NDVI')
folder = 'EVI_pf_monthly_brdf'

projection = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection()

# S1.1. eurasia - 月度聚合
start_year = 2000#2001
end_year = 2000#2023

for year in range(start_year, end_year + 1):
    for month in range(1, 13):  # 12 months per year
        # 计算月份的起始和结束日期
        start_date = datetime(year, month, 1)
        
        # 计算下一个月的第一天
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        # 转换为GEE Date格式
        start_ee = ee.Date(start_date.strftime('%Y-%m-%d'))
        end_ee = ee.Date(end_date.strftime('%Y-%m-%d'))
        
        EVI = dataset.filterDate(start_ee, end_ee).select('NDVI').median()
        EVI = EVI.rename('EVI').setDefaultProjection(crs='EPSG:4326')

        # Reproject and reduce resolution
        rescaledEVI = EVI.reproject(crs='EPSG:4326', scale=5000)\
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)

        clippedImage = rescaledEVI.clip(shapefile)
        start_date_str = start_ee.format('YYYY_MM').getInfo()
        dscr = f"EVI_{start_date_str}"
        task = ee.batch.Export.image.toDrive(image=clippedImage,
                                         region=shapefile.geometry().bounds(),
                                         description=dscr,
                                         folder=folder,
                                         fileNamePrefix=dscr,
                                         scale=5000)
        task.start()
        print(dscr)