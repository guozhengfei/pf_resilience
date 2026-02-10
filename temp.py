import numpy as np
import pymannkendall as mk
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import tifffile as tf
import os


data = tf.imread('/Volumes/Zhengfei_02/MCD43C4/MCD43C4_monthly_with_snow/NDVI/2001/MCD43C4.A2001.01.NDVI.Monthly.tif').astype(float)
data = tf.imread('/Users/zhengfei/Downloads/EVI_2001_01.tif').astype(float)
# data[data<-5000]=np.nan
# data[data>10000]=np.nan
plt.figure();plt.imshow(data)
