import entropy_estimators as ee
from scipy import stats
from os import listdir
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.cm as cm
import numpy as np
import matplotlib.patches as mpatches
import scipy.stats as sp
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib





def main():
    sns.set_style('white')
    folder1 = 'Fig3_SourceData_1'
    names1 = listdir(folder1)

    if '.DS_Store' in names1:
        names1.remove('.DS_Store')
    names1 = sorted(names1, key=lambda x: int(x.split('.')[0]))
    names1 = [folder1 + '/' + name for name in names1]


    folder2 = 'Fig3_SourceData_2'
    names2 = listdir(folder2)

    if '.DS_Store' in names2:
        names2.remove('.DS_Store')
    names2 = sorted(names2, key=lambda x: int(x.split('.')[0]))
    names2 = [folder2 + '/' + name for name in names2]

    names=names1+names2

    df_mar = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'Information': [], 'Correlation': [], 'nCells': [], 'label': []})
    tags = ['TX', 'TL']
    n = 0
    num_samps_each = 5
    total_x_wt = np.array([])
    total_x_c = np.array([])

    total_y_wt = np.array([])
    total_z_wt = np.array([])
    total_y_c = np.array([])
    total_z_c = np.array([])

    for j, name in enumerate(names):
        print(name)
        mat_contents = sio.loadmat(name)
        color_mat = mat_contents['data3D']
        t_len = mat_contents['data3D'].shape[2]

        for t in range(t_len):
            x = color_mat[:, 5, t][np.where(~np.isnan(color_mat[:, 5, t]))]
            y = color_mat[:, 7, t][np.where(~np.isnan(color_mat[:, 5, t]))]

            z = color_mat[:, 9, t][np.where(~np.isnan(color_mat[:, 5, t]))]
            spearman = sp.spearmanr(color_mat[:, 7, t][np.where(~np.isnan(color_mat[:, 5, t]))],
                                    color_mat[:, 9, t][np.where(~np.isnan(color_mat[:, 5, t]))])[0]

            if n < 2:
                if n == 0:
                    if t == t_len - 1:
                        total_x_wt = np.append(total_x_wt, x.ravel())
                        total_y_wt = np.append(total_y_wt, y.ravel())
                        total_z_wt = np.append(total_z_wt, z.ravel())
                if n == 1:
                    if t == t_len - 1:
                        total_x_c = np.append(total_x_c, x.ravel())
                        total_y_c = np.append(total_y_c, y.ravel())
                        total_z_c = np.append(total_z_c, z.ravel())

            try:
                info = ee.mi(ee.vectorize(y), ee.vectorize(z), k=3)

            except:
                info = 0

            n_cells = len(np.where(~np.isnan(color_mat[:, 5, t]))[0])
            df2 = pd.DataFrame(
                {'X': [np.std(x) / np.mean(x)], 'Y': [np.std(y) / np.mean(y)], 'Z': [np.std(z) / np.mean(z)],
                 'Information': [info], 'Correlation': [spearman], 'nCells': [n_cells], 'label': [tags[n]]})
            frames = [df_mar, df2]
            df_mar = pd.concat(frames)
        if (j + 1) % num_samps_each == 0:
            n += 1

    f3, ax3 = plt.subplots(1, 2, figsize=(10, 3))

    ax3[0].set_xlim([-10, 300])
    sns.set_style('white')
    colors = ['salmon', 'darkblue']
    tags = ['TX', 'TL']
    tvect = np.linspace(0, 400, 1000)
    recs = []
    ax30 = ax3[0].twinx()
    for j, color in enumerate(colors):
        x, y = df_mar.loc[df_mar.label == tags[j]].nCells, df_mar.loc[df_mar.label == tags[j]].X
        x, y2 = df_mar.loc[df_mar.label == tags[j]].nCells, df_mar.loc[df_mar.label == tags[j]].Information
        ys = df_mar.loc[df_mar.label == tags[j]].Y
        zs = df_mar.loc[df_mar.label == tags[j]].Z

        nbins = 15
        bins = np.linspace(0, 275, nbins)
        idx = np.digitize(x, bins)
        means = [0]
        errors = [0]
        means2 = []
        errors2 = []
        for i in range(nbins):
            if j == 0:
                ax3[0].errorbar(bins[i] + (200 / (2 * nbins)), np.mean(y[idx == i + 1]), fmt='o', color=color)

                ax3[1].errorbar(bins[i] + (200 / (2 * nbins)), np.mean(y2[idx == i + 1]), fmt='o', color=color)
            else:
                ax30.errorbar(bins[i] + (200 / (2 * nbins)), np.mean(y[idx == i + 1]), fmt='o', color=color)

                ax3[1].errorbar(bins[i] + (200 / (2 * nbins)), np.mean(y2[idx == i + 1]), fmt='o', color=color)

                #           ax.errorbar(bins[i]+(200/(2*nbins)),np.mean(y[idx==i+1]),yerr=np.std(y[idx==i+1]),fmt='o',color=color)
            means.append(np.mean(y[idx == i + 1]))
            errors.append(sp.sem(y[idx == i + 1]))
            means2.append(np.mean(y2[idx == i + 1]))
            errors2.append(sp.sem(y2[idx == i + 1]))

        means = np.asarray(means)
        errors = np.asarray(errors)

        xvect = bins + (200 / (2 * nbins))
        xvect = np.insert(xvect, 0, 0)

        if j == 0:
            ax3[0].fill_between(xvect, means - errors, means + errors, color=color, alpha=0.5)
        else:
            ax30.fill_between(xvect, means - errors, means + errors, color=color, alpha=0.5)

        means2 = np.asarray(means2)
        errors2 = np.asarray(errors2)
        ax3[1].fill_between(bins + (200 / (2 * nbins)), means2 - errors2, means2 + errors2, color=color, alpha=0.5)
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[j]))

    ax3[0].legend(recs[::-1], ['MarA Fusion', 'WT MarA'], title='Strain', loc=4)
    ax3[1].legend(recs[::-1], ['MarA Fusion', 'WT MarA'], title='Strain', loc=4)

    ax3[0].set_ylabel('Coefficient of Variation')
    ax3[0].set_title('Activator Variance over Time')
    ax3[0].set_xlabel('Number of Cells in Microcolony')
    ax3[1].set_xlabel('Number of Cells in Microcolony')

    ax3[1].set_title('Downstream coordination over time')
    ax3[1].set_ylabel('Infromation (bits)')
    ax3[0].set_ylim([0, 0.25])
    plt.tight_layout()
    f3.savefig('figures/modified_marA_TS.pdf', bbox_inches='tight')

def Simplified_simulation(t_int,t_end,N,tau_x,tau_y,tau_z,gy,gz):
    dt = float(t_end - t_int) / N

    ### Initialize Matrix for all possible cells at all time points for all dimension of data (lets say we're not going to let it get more than 100)
    Intrinsic_Noise = np.empty((N, 3))
    Intrinsic_Noise[:,:]= np.NAN

    Gene = np.empty((N, 3))
    Gene[:,:]= np.NAN


    g_y=gy
    g_z=gz

    lambda_x=(1.0 - np.exp(-dt / tau_x))

    ### initializing values

    cellular_indexes = np.array(range(100))
    # for index in cellular_indexes:
    Gene[0, 0] = 0
    Gene[0, 1] = 0
    Gene[0, 2] = 0

    for t in np.arange(N-1)+1:


        Gene[t,0] = Gene[t - 1,0] + dt * (-Gene[t - 1,0]/tau_x)+np.sqrt(2/tau_x)*np.random.normal(scale=np.sqrt(dt),loc=0.0)
        Gene[t,1] = Gene[t - 1,1] + dt * (g_y*Gene[t - 1,0]-Gene[t - 1,1]/tau_y)+np.sqrt(2/tau_y)*np.random.normal(loc=0.0,scale=np.sqrt(dt))
        Gene[t, 2] = Gene[t - 1,2] + dt * (g_z*Gene[t - 1,0]-Gene[t - 1,2]/tau_z)+np.sqrt(2/tau_z)*np.random.normal(loc=0.0,scale=np.sqrt(dt))

    return Gene

def traces():

    fig,ax=plt.subplots(1,2, gridspec_kw = {'width_ratios':[3, 1]},figsize=(12,4))
    t_end = 100000
    t_int = 0
    N = 400000
    dt = float(t_end - t_int) / N
    time_vect = np.arange(t_int, t_end, dt)
    gy = 0.1
    gz = 0.1
    tau_x = 10

    tau_y = 30.1 / np.log(2)

    tau_z = 30.2 / np.log(2)


    colors = cm.rainbow(np.linspace(0, 1, 18))[[1,5,10]]
    tauxs=[1,5,10]
    for l in range(3):
        tau_x = tauxs[l]

        Gene = Simplified_simulation(t_int, t_end, N, tau_x / np.log(2), tau_y, tau_z, gy, gz)
        ax[0].plot(time_vect, Gene[:, 0], linewidth=3, color=colors[l])
        ax[0].set_ylim([-5,5])
        ax[0].set_xlim([0,100])


        ax[1].hist(Gene[:, 0],bins=25, normed=1,orientation="horizontal",histtype = 'step',color=colors[l],linewidth=3)
        ax[1].set_ylim([-5,5])
    ax[0].set_xlabel('Time (Minutes')
    ax[0].set_ylabel('Relative Protein Concentration')
    ax[1].set_ylabel('Relative Protein Concentration')
    ax[1].set_xlabel('Probablity')
    plt.xticks([0,.25])

    fig.savefig('figures/modified_marA_simulation_traces.pdf')


if __name__ == "__main__":
    main()
    traces()