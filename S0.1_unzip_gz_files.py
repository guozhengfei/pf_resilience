import gzip
import shutil
import os

def unzip_gz_file(gz_file_path, output_path):
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_path, 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)

# Example usage
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
gz_files = os.listdir(current_dir+'/1_Input/GPP_SIF/')

for file in gz_files:
    gz_file_path = current_dir+'/1_Input/GPP_SIF/'+file  # Replace with the actual path to your .gz file
    output_path = current_dir+'/1_Input/GPP_SIF/'+file.split('.')[0]+file.split('.')[1]+'.tif'  # Replace with the desired output path for the unzipped file
    print(file)

    unzip_gz_file(gz_file_path, output_path)