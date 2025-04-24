import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import tifffile as tf
from plot_NH import *
import os
from PIL import Image
import rasterio
import cv2

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    break_points = np.load(current_dir+'/2_Output/break_points_modis.npy')+2002
    plt.figure(figsize=(1.9,1.5)); plt.hist(break_points,bins=20,ec='k')
    plt.tight_layout()
    # figToPath = current_dir + '/4_Figures/Fig01_break_point_hist'
    # plt.savefig(figToPath, dpi=600)

