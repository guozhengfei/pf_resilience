import os
import shutil


def compare_and_move_folders(folder1, folder2, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取两个文件夹的文件名（忽略隐藏文件）
    files1 = {f[:16]: f for f in os.listdir(folder1) if not f.startswith('.')}
    files2 = {f[:16]: f for f in os.listdir(folder2) if not f.startswith('.')}

    # 找出文件名前10个字符不同的文件
    moved_count = 0

    # 检查folder1中的文件
    for prefix, filename in files1.items():
        if prefix not in files2:
            source_path = os.path.join(folder1, filename)
            dest_path = os.path.join(output_folder, filename)
            shutil.move(source_path, dest_path)
            print(f"移动: {filename} -> {output_folder}")
            moved_count += 1

    # 检查folder2中的文件
    for prefix, filename in files2.items():
        if prefix not in files1:
            source_path = os.path.join(folder2, filename)
            dest_path = os.path.join(output_folder, filename)
            shutil.move(source_path, dest_path)
            print(f"移动: {filename} -> {output_folder}")
            moved_count += 1

    print(f"完成！共移动了 {moved_count} 个文件")


# 使用方法
folder1 = "/Volumes/Zhengfei_02/MCD43C4/MCD43C4_daily_tif/NDVI"
folder2 = "/Volumes/Zhengfei_02/MCD43C4/MCD43C4_061-20251112_085746"
output_folder = "/Volumes/Zhengfei_02/MCD43C4/MCD43C4_061-20251112_085745"

compare_and_move_folders(folder1, folder2, output_folder)