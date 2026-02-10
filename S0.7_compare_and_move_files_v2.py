import os
import numpy as np
import rasterio


def average_adjacent_geotiff(source_folder, target_folder):
    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有tif文件并按文件名排序
    tif_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]
    tif_files.sort()

    processed_count = 0

    # 每两个文件为一组进行处理
    for i in range(0, len(tif_files), 2):
        # 如果是最后一个文件且没有配对，则跳过
        if i + 1 >= len(tif_files):
            print(f"跳过: {tif_files[i]} (没有配对的文件)")
            break

        file1 = tif_files[i]
        file2 = tif_files[i + 1]

        try:
            # 读取两个tif文件
            with rasterio.open(os.path.join(source_folder, file1)) as src1:
                data1 = src1.read()
                profile = src1.profile

            with rasterio.open(os.path.join(source_folder, file2)) as src2:
                data2 = src2.read()

            # 计算平均值
            averaged_data = (data1.astype(np.float32) + data2.astype(np.float32)) / 2.0

            # 生成输出文件名
            output_name = file1
            output_path = os.path.join(target_folder, output_name)

            # 保存结果
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(averaged_data.astype(profile['dtype']))

            print(f"处理: {file1} + {file2} -> {output_name}")
            processed_count += 1

        except Exception as e:
            print(f"处理 {file1} 和 {file2} 时出错: {e}")

    print(f"完成！原文件数: {len(tif_files)}, 新文件数: {processed_count}")


# 使用方法
source_folder = "/Volumes/Zhengfei_01/GLASS_NDVI_005_TIF"
target_folder = "/Volumes/Zhengfei_01/GLASS_NDVI_005_16d"

average_adjacent_geotiff(source_folder, target_folder)