import os
import shutil


def copy_all_files(source_folder, target_folder):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    copied_count = 0

    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 源文件路径
            source_path = os.path.join(root, file)

            # 目标文件路径（直接放在目标文件夹，不保留子目录结构）
            target_path = os.path.join(target_folder, file)

            # 如果目标文件已存在，重命名避免冲突
            counter = 1
            original_target = target_path
            while os.path.exists(target_path):
                name, ext = os.path.splitext(file)
                target_path = os.path.join(target_folder, f"{name}_{counter}{ext}")
                counter += 1

            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f"复制: {source_path} -> {target_path}")
            copied_count += 1

    print(f"完成！共复制了 {copied_count} 个文件")


# 使用方法
source_folder = "/Volumes/Zhengfei_02/MCD43C4/MCD43C4_monthly_with_snow/NDVI"
target_folder = "/Volumes/Zhengfei_01/Project 2 pf resilience/1_Input/MCD43C4_monthly_ndvi"

copy_all_files(source_folder, target_folder)