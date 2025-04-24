import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    directory = current_dir + "/1_Input/Fire/hotspotFRPSum_20240229_TERRA/"
    specific_str = "FRP"
    frp_files = [file for file in os.listdir(directory) if file.endswith(".csv") and specific_str in file]
    grid_files = [file for file in os.listdir(directory) if file.endswith(".csv") and 'grid' in file]

    grid_data = pd.read_csv(directory+grid_files[0])

    frp_glb_sum = []
    threshold = 5000
    for name in frp_files:
        data = pd.read_csv(directory+name)
        data = data.loc[data['FRP_sum']>threshold,:]
        frp_yearly = data.groupby('ID', as_index=False)['FRP_sum'].sum()
        frp_glb_sum.append(frp_yearly['FRP_sum'].sum())
        # frp_glb_sum.append(data.shape[0])

    np.percentile(data['FRP_sum'],99)
    plt.figure(); plt.plot(frp_glb_sum)

