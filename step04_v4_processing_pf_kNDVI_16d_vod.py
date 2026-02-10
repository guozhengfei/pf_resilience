import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import os
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

def ar1_series_4yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 23

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, center=True).apply(calc_ar1).values
    return ar1

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'

    # open pf mask as dataset to keep georeference (transform/crs/size)
    pf_mask_ds = rasterio.open(pf_mask_path2)
    pf_mask2 = pf_mask_ds.read(1).astype(float)

    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan

    # create the pf_mask used later (downsampled and flipped) to match original pipeline
    pf_mask = pf_mask2[::3, ::3]

    #plt.figure();plt.imshow(pf_mask)

    EVI_folder = current_dir + '/1_Input/vod_16day_month_aggregate/16day_aggregate/'
    filenames = os.listdir(EVI_folder)
    evi_filenames = []
    for name in filenames:
        if name.startswith('.'): continue
        evi_filenames.append(name)
    evi_filenames = np.sort(evi_filenames)  # [:529]

    EVI = []
    yr_num = 22
    bands_year = 23

    # prepare target geometry from pf_mask_ds
    dst_height = pf_mask_ds.height
    dst_width = pf_mask_ds.width
    dst_transform = pf_mask_ds.transform
    dst_crs = pf_mask_ds.crs

    # show progress for reproject/extract stage
    print(f"开始重投影并提取 {len(evi_filenames)} 个文件到目标格网...")
    for name in tqdm(evi_filenames, desc='Reproject/extract', unit='file'):
        src_path = os.path.join(EVI_folder, name)
        with rasterio.open(src_path) as src:
            # allocate destination array matching pf_mask2 georeference/size
            dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )

        evi = dst
        evi_ds = evi[::3, ::3]     # match downsample factor used for pf_mask

        # extract values using the already prepared pf_mask (downsampled/flipped)
        evi_val = evi_ds[~np.isnan(pf_mask)]
        EVI.append(evi_val)

    # close pf_mask dataset
    pf_mask_ds.close()

    EVI = np.array(EVI).T

    EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
    EVI_sg[EVI_sg < 0] = 0
    ser = EVI_sg.T
    del EVI
    EVI_yr = np.zeros_like(ser)

    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr

    Evi_sea_rep = np.zeros_like(rm_offline)
    for yr in range(yr_num):
        start_index = (yr - 5) * bands_year
        end_index = (yr + 5) * bands_year
        if yr < 5: start_index = 0; end_index = 10 * bands_year
        if yr > (yr_num - 5): start_index = (yr_num - 10) * bands_year; end_index = yr_num * bands_year
        data_i = rm_offline[:, start_index:end_index]
        Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / 23), bands_year)), axis=1)
        Evi_sea_rep[:, yr * 23:(yr + 1) * 23] = Evi_sea
    res = rm_offline - Evi_sea_rep
    res[np.isnan(res)]=0
    # res = res[::100,:]
    # plt.figure(); plt.plot(np.nanmean(res, axis=0))

    del rm_offline, Evi_sea_rep
    divided_arrays = [row for row in res]

    # compute AR1 with progress bar
    print(f"开始计算 AR1（每像元） 共 {len(divided_arrays)} 个像元，使用进程数 {min(6, mp.cpu_count() - 1)} ...")
    with mp.Pool(min(6, mp.cpu_count() - 1)) as pool:
        results = []
        for r in tqdm(pool.imap(ar1_series_4yr, divided_arrays), total=len(divided_arrays), desc='AR1 per-pixel', unit='pix'):
            results.append(r)
    ar1_res_5sg = np.array(results)

    # plt.figure(); plt.plot(np.linspace(2000,2021+22/23,506),np.nanmean(ar1_res_5sg,axis=0))
    output_file = current_dir + '/2_Output/spatial_resilience/ar1_5yr_vod_sg_rolling.npy'
    np.save(output_file, ar1_res_5sg)
    print(f"保存完成: {output_file}")
