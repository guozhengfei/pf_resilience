import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
from PIL import Image
import multiprocess as mp
import os
def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2
    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))
    return smoothed_arr

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    files = os.listdir(current_dir +'/2_Output/Trendy_resilience_v2/')
    file_list = []
    for file in files:
        if not file.startswith('.'):
            file_list.append(file)

    TAC = [];
    Trend = []
    for name in file_list:
        tac = np.load(current_dir +'/2_Output/Trendy_resilience_v2/'+name)
        tac = tac[:,24:-24]
        mask = np.isnan(tac).any(axis=1)
        tac = tac[~mask,:]
        tac08 = tac[:,:84]
        tac23 = tac[:,84:]
        tac_trend_08 = np.polyfit(np.arange(tac08.shape[1]) / 12, tac08.T, deg=1)[0,:]
        tac_trend_23 = np.polyfit(np.arange(tac23.shape[1]) / 12, tac23.T, deg=1)[0, :]
        trend08_mean = tac_trend_08.mean()
        trend08_sd = tac_trend_08.std()
        trend23_mean = tac_trend_23.mean()
        trend23_sd = tac_trend_23.std()
        tac_mean = np.nanmean(tac,axis=0)
        Trend.append([trend08_mean,trend08_sd, trend23_mean,trend23_sd])
        TAC.append([tac_mean])
        # if np.nanmean(tac_mean)>0:
        #     axs[0].plot(time_label_modis,tac_mean,lw=0.8)

    Trend = np.array(Trend)
    TAC = np.array(TAC)
    TAC_mean = np.nanmean(TAC,axis=2).reshape(-1)

    TAC = TAC[TAC_mean>0,:,:] # remove the tac <0 model results
    Trend = Trend[TAC_mean>0,:]

    from brokenaxes import brokenaxes

    fig, axs = plt.subplots(1, figsize=(12 * 0.8, 2.5 * 0.8))
    bax = brokenaxes(ylims=((-0.025, -0.02),(-0.015, 0.015), (0.040, 0.047)), hspace=0.05)
    bax.bar(range(15),Trend[:,0],yerr=Trend[:,1]*0.1,width=0.3,color='#d6604d', ec='k', hatch='//')
    bax.bar(15, Trend[:, 0].mean(), yerr=Trend[:,1].mean()*0.1,width=0.3, color='#4393c3', ec='k', hatch='//')
    bax.bar(np.array(range(15))+0.3, Trend[:, 2],yerr=Trend[:,3]*0.1, width=0.3,color='#d6604d', ec='k')
    bax.bar(15+0.3, Trend[:, 2].mean(), yerr=Trend[:,3].mean()*0.1,width=0.3, color='#4393c3', ec='k', hatch='//')
    axs.set_axis_off()

    xtick=[]
    for file in file_list:
        xtick.append(file.split('.')[0].split('_')[0])
    xtick.pop(-5)
    xtick = xtick + ['All']
    bax.set_xticks(range(16),xtick,rotation=90)
    # fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig05_resilience_trendy01'
    plt.savefig(figToPath, dpi=600)

    time_label_modis = np.linspace(2002, 2020, 19)
    fig, axs = plt.subplots(4, 4, figsize=(12*0.8, 7*0.8),sharex=True)
    i = 0
    for row in range(4):
        for col in range(4):
            if i>14:
                tac_i = np.mean(TAC,axis=0).reshape(-1)
                cl = '#4393c3'
            else:
                tac_i = TAC[i,:,:].reshape(-1)
                cl = '#d6604d'
            tac_i_rsp = tac_i.reshape(19,12)
            tac_i_yr = np.mean(tac_i_rsp,axis=1)
            sd = np.std(tac_i_rsp,axis=1)
            # axs[row,col].plot(time_label_modis,tac_i_yr-tac_i_yr[0],color = cl,lw=2)
            axs[row,col].plot(time_label_modis,tac_i_yr,color = cl,lw=2)
            # axs[row, col].fill_between(time_label_modis,tac_i_yr-tac_i_yr[0]+sd,tac_i_yr-tac_i_yr[0]-sd,color = cl,alpha=0.3)
            axs[row, col].fill_between(time_label_modis,tac_i_yr+sd,tac_i_yr-sd,color = cl,alpha=0.3)

            axs[row, col].set_title(xtick[i])
            i = i + 1

        fig.tight_layout()
        figToPath = current_dir + '/4_Figures/Fig05_resilience_trendy02'
        plt.savefig(figToPath, dpi=600)




