from pathlib import Path
import zipfile


DATA_DIR = Path('/Volumes/Zhengfei_02/Fluxnet_2025')


# Step 1: remove trailing .jpg or .jpg_ from real downloaded zip files
for file_path in sorted(DATA_DIR.iterdir()):
    if not file_path.is_file() or file_path.name.startswith('._'):
        continue

    new_name = file_path.name
    if new_name.endswith('.jpg_'):
        new_name = new_name[:-5]
    elif new_name.endswith('.jpg'):
        new_name = new_name[:-4]
    else:
        continue

    new_path = file_path.with_name(new_name)
    if new_path.exists():
        print(f'Skip rename, exists: {new_path.name}')
        continue

    file_path.rename(new_path)
    print(f'Renamed: {file_path.name} -> {new_path.name}')


# Step 2: unzip renamed files
zip_files = [p for p in sorted(DATA_DIR.iterdir())
             if p.is_file() and not p.name.startswith('._') and '.zip' in p.name]

for zip_file in zip_files:
    extract_dir = DATA_DIR / zip_file.name.split('.zip')[0]
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(extract_dir)
    print(f'Unzipped: {zip_file.name} -> {extract_dir.name}/')

print('All files processed.')
