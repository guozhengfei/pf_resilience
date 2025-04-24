import rasterio
import matplotlib.pyplot as plt
import matplotlib;# matplotlib.use('Qt5Agg')
import numpy as np
import cartopy.crs as ccrs
import matplotlib.path as mpath

def plot_NH(data):
    file_path = r'D:\Projects\Project_pf\Data\EVI_pf_16d\EVI_550.tif'
    dataset = rasterio.open(file_path)
    # data = dataset.read(1)
    # plt.figure();plt.imshow(data)
    # Read the corresponding longitude and latitude arrays
    left,bottom,right,top = np.squeeze(dataset.bounds)
    latitudes = np.linspace(top,bottom,dataset.height)
    longitudes = np.linspace(left,right,dataset.width)

    # Create a 2D grid using meshgrid
    grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo())
    # ax1.contourf(grid_longitudes[::10,::10], grid_latitudes[::10,::10], data[::10,::10], transform=ccrs.PlateCarree())
    this = ax1.pcolormesh(grid_longitudes[::10,::10], grid_latitudes[::10,::10], data[::10,::10], transform=ccrs.PlateCarree())

    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this,orientation='horizontal',label='ar1',fraction=0.03,pad=0.05)
    # ax1.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    return ax1
