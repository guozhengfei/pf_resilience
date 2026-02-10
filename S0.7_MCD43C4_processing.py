import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_bounds
from scipy import stats
from pyhdf.SD import SD, SDC
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import psutil

warnings.filterwarnings('ignore')


class MCD43C4NDVIProcessor:
    def __init__(self, data_dir='input_data', output_dir='output_monthly_ndvi', processing_mode='yearly', n_workers=None):
        """
        初始化MCD43C4 NDVI处理器

        Parameters:
        data_dir: 输入HDF文件目录
        output_dir: 输出NDVI TIFF文件目录
        processing_mode: 处理模式 ('yearly' 或 'monthly')
        n_workers: 并行工作进程数（None为自动）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processing_mode = processing_mode
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用 {self.n_workers} 个工作进程")

    def parse_date_from_filename(self, filename):
        """
        从文件名解析日期
        """
        try:
            basename = os.path.basename(filename)
            parts = basename.split('.')
            if len(parts) < 2:
                return None

            date_str = parts[1][1:]  # 提取AYYYYDDD格式的日期
            if len(date_str) != 7:
                return None

            year = int(date_str[:4])
            doy = int(date_str[4:7])

            if year < 2000 or year > 2030 or doy < 1 or doy > 366:
                return None

            date = datetime(year, 1, 1) + timedelta(doy - 1)
            return date
        except Exception as e:
            return None

    def get_year_month_from_filename(self, filename):
        """
        从文件名获取年份和月份
        """
        date_obj = self.parse_date_from_filename(filename)
        if date_obj:
            return date_obj.year, date_obj.month
        return None, None

    def group_files_by_period(self, hdf_files):
        """
        按年份或月份分组文件
        """
        if self.processing_mode == 'yearly':
            return self._group_files_by_year(hdf_files)
        else:  # monthly
            return self._group_files_by_month(hdf_files)

    def _group_files_by_year(self, hdf_files):
        """
        按年份分组文件
        """
        yearly_groups = {}
        for hdf_file in hdf_files:
            year, _ = self.get_year_month_from_filename(hdf_file)
            if year is not None:
                if year not in yearly_groups:
                    yearly_groups[year] = []
                yearly_groups[year].append(hdf_file)

        # 对每个年份的文件按日期排序
        for year in yearly_groups:
            yearly_groups[year].sort(key=lambda x: self.parse_date_from_filename(x))

        return yearly_groups

    def _group_files_by_month(self, hdf_files):
        """
        按年月分组文件
        """
        monthly_groups = {}
        for hdf_file in hdf_files:
            year, month = self.get_year_month_from_filename(hdf_file)
            if year is not None and month is not None:
                year_month = f"{year}-{month:02d}"
                if year_month not in monthly_groups:
                    monthly_groups[year_month] = []
                monthly_groups[year_month].append(hdf_file)

        # 对每个月的文件按日期排序
        for year_month in monthly_groups:
            monthly_groups[year_month].sort(key=lambda x: self.parse_date_from_filename(x))

        return monthly_groups

    def calculate_ndvi(self, hdf_file):
        """
        使用pyhdf计算Kernel NDVI
        """
        try:
            hdf = SD(hdf_file, SDC.READ)

            # 读取红光波段(Band1)
            red_sds = hdf.select('Nadir_Reflectance_Band1')
            red_band = red_sds.get().astype(np.float32)
            red_attrs = red_sds.attributes()

            # 读取近红外波段(Band2)
            nir_sds = hdf.select('Nadir_Reflectance_Band2')
            nir_band = nir_sds.get().astype(np.float32)
            nir_attrs = nir_sds.attributes()

            # 读取percent_input layer
            p_input = hdf.select('Percent_Inputs')
            p_input_band = p_input.get().astype(np.float32)
            p_input_band[(p_input_band==255)| (p_input_band<20)]=0

            # 读取percent_snow layer
            p_snow = hdf.select('Percent_Snow')
            p_snow_band = p_snow.get().astype(np.float32)

            # 获取填充值
            fill_value = -9999.0
            if '_FillValue' in red_attrs:
                fill_value = float(red_attrs['_FillValue'])

            # 获取缩放因子
            scale_factor = 1.0
            if 'scale_factor' in red_attrs:
                scale_factor = float(red_attrs['scale_factor'])

            # 应用缩放因子
            if scale_factor != 1.0:
                red_band = red_band * scale_factor
                nir_band = nir_band * scale_factor

            # 关闭数据集
            red_sds.endaccess()
            nir_sds.endaccess()
            hdf.end()

            # 检查数据维度
            if red_band.shape != nir_band.shape:
                return None, None

            # 计算NDVI - 向量化操作，避免循环
            denominator = nir_band + red_band
            ndvi = np.full_like(red_band, fill_value, dtype=np.float32)

            valid_mask = (denominator != 0) & (red_band != fill_value) & (nir_band != fill_value) & (p_input_band != 0) & (p_snow_band == 0)
            ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
            ndvi[~valid_mask] = 0

            # NDVI值域检查
            ndvi = np.clip(ndvi, -1.0, 1.0)

            return ndvi, fill_value

        except Exception as e:
            print(f"计算NDVI时出错 {hdf_file}: {e}")
            try:
                hdf.end()
            except:
                pass
            return None, None

    @staticmethod
    def _process_single_file(hdf_file):
        """
        静态方法用于并行处理单个文件
        """
        processor = MCD43C4NDVIProcessor.__new__(MCD43C4NDVIProcessor)
        ndvi, fill_value = processor.calculate_ndvi(hdf_file)
        if ndvi is not None:
            return (hdf_file, ndvi, fill_value)
        return None

    def process_time_series_for_period_parallel(self, file_group, period_name):
        """
        使用并行处理单个时间段的时间序列数据
        """
        print(f"\n处理时间段: {period_name}")
        print(f"文件数量: {len(file_group)}")

        if len(file_group) == 0:
            print(f"时间段 {period_name} 没有文件")
            return None

        # 获取第一个文件的维度
        first_result = self._process_single_file(file_group[0])
        if first_result is None:
            print(f"无法读取时间段 {period_name} 的第一个文件")
            return None

        _, first_ndvi, fill_value = first_result
        n_lat, n_lon = first_ndvi.shape
        print(f"数据维度: 纬度={n_lat}, 经度={n_lon}")

        # 并行处理所有文件
        print(f"使用 {self.n_workers} 个进程并行处理...")
        results = []
        with Pool(processes=self.n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(self._process_single_file, file_group, chunksize=5)):
                if result is not None:
                    results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"进度: {i + 1}/{len(file_group)}")

        if not results:
            print(f"时间段 {period_name} 没有成功处理任何文件")
            return None

        print(f"成功处理 {len(results)} 个文件")

        # 构建3D数组
        all_ndvi = np.zeros((len(results), n_lat, n_lon), dtype=np.float32)
        file_list = []
        for idx, (hdf_file, ndvi, _) in enumerate(results):
            all_ndvi[idx] = ndvi
            file_list.append(hdf_file)

        return all_ndvi, fill_value, file_list

    def aggregate_to_monthly(self, file_group, ndvi_3d, fill_value):
        """
        聚合到月分辨率 - 优化版本
        """
        monthly_ndvi = {}

        for time_idx, hdf_file in enumerate(file_group):
            date_obj = self.parse_date_from_filename(hdf_file)
            if date_obj is None:
                continue

            year_month = date_obj.strftime('%Y-%m')
            if year_month not in monthly_ndvi:
                monthly_ndvi[year_month] = []

            daily_ndvi = ndvi_3d[time_idx].copy()
            daily_ndvi[daily_ndvi == 0] = np.nan
            valid_mask = (daily_ndvi != fill_value) & (~np.isnan(daily_ndvi)) & (daily_ndvi >= -1.0) & (daily_ndvi <= 1.0)

            if np.sum(valid_mask) > 0:
                monthly_ndvi[year_month].append(daily_ndvi)

        # 计算每月最大值 - 使用向量化操作
        monthly_results = {}
        for year_month, daily_data_list in monthly_ndvi.items():
            if daily_data_list:
                monthly_stack = np.stack(daily_data_list, axis=0)
                monthly_max = np.nanmax(monthly_stack, axis=0)
                monthly_max[~np.isfinite(monthly_max)] = fill_value
                monthly_max[(monthly_max < -1.0) | (monthly_max > 1.0)] = fill_value
                monthly_results[year_month] = monthly_max
                print(f"月份 {year_month}: 聚合了 {len(daily_data_list)} 天数据")

        return monthly_results

    def save_ndvi_as_geotiff(self, ndvi_data, output_file, fill_value=-9999.0):
        """
        保存NDVI数据为GeoTIFF格式
        """
        try:
            transform = from_bounds(-180, -90, 180, 90, ndvi_data.shape[1], ndvi_data.shape[0])

            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': fill_value,
                'width': ndvi_data.shape[1],
                'height': ndvi_data.shape[0],
                'count': 1,
                'crs': 'EPSG:4326',
                'transform': transform,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'predictor': 2
            }

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(ndvi_data.astype(np.float32), 1)

            print(f"成功保存: {output_file}")

        except Exception as e:
            print(f"保存GeoTIFF文件失败 {output_file}: {e}")

    def process_ndvi_time_series(self):
        """
        主处理函数，按年份或月份处理数据
        """
        print("开始处理MCD43C4 NDVI时间序列数据...")
        print(f"处理模式: {self.processing_mode}")
        print(f"CPU核心数: {cpu_count()}")
        print(f"使用工作进程: {self.n_workers}")

        # 查找所有HDF文件
        hdf_files = glob.glob(os.path.join(self.data_dir, '*.hdf'))
        if not hdf_files:
            print(f"在目录 {self.data_dir} 中未找到HDF文件")
            return

        print(f"找到 {len(hdf_files)} 个HDF文件")
        
        # 按时间段分组文件
        period_groups = self.group_files_by_period(hdf_files)

        if not period_groups:
            print("没有找到有效的时间段分组")
            return

        print(f"找到 {len(period_groups)} 个时间段需要处理")

        # 按时间段处理数据
        for period_name, file_group in period_groups.items():
            print(f"\n{'=' * 50}")
            print(f"开始处理时间段: {period_name}")
            print(f"{'=' * 50}")

            # 并行处理当前时间段
            result = self.process_time_series_for_period_parallel(file_group, period_name)
            if result is None:
                print(f"时间段 {period_name} 处理失败")
                continue

            filtered_ndvi, fill_value, file_list = result

            # 聚合到月分辨率
            print("聚合到月分辨率...")
            monthly_ndvi_data = self.aggregate_to_monthly(file_list, filtered_ndvi, fill_value)

            if not monthly_ndvi_data:
                print(f"时间段 {period_name} 没有生成月度数据")
                continue

            # 保存月NDVI数据
            print("保存月NDVI数据...")
            for year_month, ndvi_data in monthly_ndvi_data.items():
                output_file = os.path.join(self.output_dir, f'MCD43C4_NDVI_monthly_{year_month}.tif')
                self.save_ndvi_as_geotiff(ndvi_data, output_file, fill_value)

            print(f"时间段 {period_name} 处理完成")

            # 释放内存
            del filtered_ndvi
            del monthly_ndvi_data

        print("\n所有时间段处理完成！")


def main():
    """主函数"""
    processor = MCD43C4NDVIProcessor(
        data_dir='/Volumes/Zhengfei_02/MCD43C4/MCD43C4_061-20251112_085746',
        output_dir='/Volumes/Zhengfei_02/MCD43C4/MCD43C4_monthly_ndvi',
        processing_mode='monthly',  # 可选 'yearly' 或 'monthly'
        n_workers=12  # 自动设置为cpu_count - 1
    )
    processor.process_ndvi_time_series()


if __name__ == "__main__":
    main()