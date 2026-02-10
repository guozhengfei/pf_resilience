import os
import glob
import numpy as np
from pyhdf import SD
import rasterio
from rasterio.transform import from_origin
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def extract_evi_to_geotiff(hdf_path, out_folder):
    """
    Extract NDVI from HDF file and save as GeoTIFF
    """
    try:
        hdf_ds = SD.SD(hdf_path)
        
        # Read Band 2 (NIR) and Band 1 (Red)
        sds_obj2 = hdf_ds.select('Nadir_Reflectance_Band2')
        nir = sds_obj2.get().astype(np.float32)
        
        sds_obj1 = hdf_ds.select('Nadir_Reflectance_Band1')
        red = sds_obj1.get().astype(np.float32)
        
        # Close HDF dataset
        hdf_ds.end()
        
        # Calculate NDVI with vectorized operations
        denominator = nir + red
        evi_array = np.full_like(red, -9999, dtype=np.int16)
        
        # Only calculate where denominator is not zero
        valid_mask = (denominator != 0) & (red != 32767) & (nir != 32767)
        evi_array[valid_mask] = ((nir[valid_mask] - red[valid_mask]) / 
                                  denominator[valid_mask] * 10000).astype(np.int16)
        
        # Get geolocation info (MODIS global grid, 0.05 deg, 3600x720 rows/cols)
        nrows, ncols = evi_array.shape
        lon_min, lat_max = -180, 90
        pixel_size = 0.05
        transform = from_origin(lon_min, lat_max, pixel_size, pixel_size)
        
        # Output file name
        base = os.path.splitext(os.path.basename(hdf_path))[0]
        out_path = os.path.join(out_folder, f"{base}_NDVI.tif")
        
        # Save as compressed GeoTIFF
        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,
            dtype=evi_array.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256,
            predictor=2  # Improves compression for integer data
        ) as dst:
            dst.write(evi_array, 1)
        
        return f"✓ {base}"
    
    except Exception as e:
        return f"✗ {os.path.basename(hdf_path)}: {str(e)}"


if __name__ == '__main__':
    # Set multiprocessing start method for macOS compatibility
    mp.set_start_method('spawn', force=True)
    
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    hdf_folder = '/Volumes/Zhengfei_02/MCD43C4/MCD43C4_061-20251112_085746'
    out_folder = '/Volumes/Zhengfei_02/MCD43C4/MCD43C4_daily_ndvi_with_snow'
    os.makedirs(out_folder, exist_ok=True)
    
    # Find all HDF files
    hdf_files = sorted(glob.glob(os.path.join(hdf_folder, '*.hdf')))[7390:]
    
    if not hdf_files:
        print(f"No HDF files found in {hdf_folder}")
        exit(1)
    
    print(f"Found {len(hdf_files)} HDF files")
    print(f"Processing with {min(mp.cpu_count() - 1, 8)} workers...\n")
    
    # Create partial function with output folder
    process_func = partial(extract_evi_to_geotiff, out_folder=out_folder)
    
    # Parallel processing with progress bar
    n_workers = max(1, min(12, mp.cpu_count() - 1))
    
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, hdf_files, chunksize=3),
            total=len(hdf_files),
            desc='Processing HDF files',
            unit='file'
        ))
    
    # Print summary
    print(f"\n{'='*60}")
    successful = sum(1 for r in results if r.startswith('✓'))
    failed = sum(1 for r in results if r.startswith('✗'))
    
    print(f"Completed: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if r.startswith('✗'):
                print(f"  {r}")
    
    print(f"\nOutput saved to: {out_folder}")