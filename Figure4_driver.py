import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
def transmitted_YZ0(tx, ty, tz, gy, gz):
    dx = 1.0
    dy = 1.0
    dz = 1.0
    return (1 / (tx)) * (np.sqrt(1.0 + ty ** 2 * gy ** 2 * ((tx * dx) / (dy * ty))) + np.sqrt(
        1 + tz ** 2 * gz ** 2 * ((tx * dx) / (dz * tz))) - np.sqrt(
        1.0 + (tx * dx) * (tz ** 2 * gz ** 2 / (dz * tz) + ty ** 2 * gy ** 2 / (dy * ty))) - 1.0)


def transmitted_YZ2(tx, ty, tz, gy, gz):
    dx = 1.0
    dy = 1.0
    dz = 1.0
    return (1 / (tx)) * (np.sqrt(1 + gy ** 2 * ty * tx) + np.sqrt(1 + gz ** 2 * tz * tx) - np.sqrt(
        1 + gz ** 2 * tz * tx + gy ** 2 * ty * tx) - 1)


def analytic_xcor_YZ(t, tx, ty, tz, gy, gz):
    #     values=np.zeros((len(t),1)).ravel()
    #     tg0=t[t>0]
    #     # values[t>0]=gy*gz*tz*ty*np.sqrt(np.pi/2)*(1/(ty**2-tz**2))*(ty*np.exp(tg0/ty)-tz*np.exp(tg0/tz))

    #     tl0 = t[t <= 0]
    # normalization factors
    R_AA = np.sqrt(np.pi / 2) * (tz / (2 * tx * ty) + ((gy ** 2 * ty ** 2) / (2 * (ty ** 2 - tx ** 2))) * (ty - tx))
    R_BB = np.sqrt(np.pi / 2) * (tz / (2 * tx * tz) + ((gz ** 2 * tz ** 2) / (2 * (tz ** 2 - tx ** 2))) * (tz - tx))
    NAB = 1 / np.sqrt(R_AA * R_BB)
    alpha = 2 * tz ** 2 * np.exp(-0 / tz) / ((ty + tz) * (tz ** 2 - tx ** 2))

    beta = tx * np.exp(-0 / tx) / ((ty + tx) * (tx - tz))

    #     values[t > 0] =.5*np.sqrt(np.pi/2)*gy*gz*ty*tz*(alpha+beta)

    alpha = ((2 * ty ** 2 / ((ty + tz) * (ty ** 2 - tx ** 2))) * np.exp(0 / ty))

    beta = (1 / ((ty - tx) * (tz + tx))) * tx * np.exp(0 / tx)

    value = .5 * gy * gz * tz * ty * np.sqrt(np.pi / 2) * (alpha - beta)
    # values[t <= 0] =-gy*gz*tz*ty*np.sqrt(np.pi/2)*((1/((ty+tx)*(tx-tz)))*tx*np.exp(tl0/tx)+np.exp(tl0/tx))

    return value * NAB


def mi_transform(val):
    return -0.5 * np.log(1 - val ** 2)
def main():
    plt.close('all')
    figx, axx = plt.subplots()



    #     k=n
    #     axx.plot(analytic_xcor_YZ(0.0,temptaux,(half_lives[k])/np.log(2),(half_lives[k]+.01)/np.log(2),0.1,0.1),transmitted_YZ2(temptaux/np.log(2),(half_lives[k])/np.log(2),(half_lives[k]+0.001)/np.log(2),0.1,0.1),linewidth=3,color=color,)
    colors = cm.rainbow(np.linspace(0, 1, 10000))
    xhls = np.linspace(0, 100, 10000)

    g = 0.1
    for n, color in enumerate(colors):
        val = analytic_xcor_YZ(0.0, xhls[n] / np.log(2), (30.0) / np.log(2), (30.0 + .01) / np.log(2),
                               g, g)
        axx.errorbar(mi_transform(val), transmitted_YZ2(xhls[n] / np.log(2), (30.0) / np.log(2),
                                                        (30.0 + .01) / np.log(2), g, g), linewidth=5,
                     color=color, fmt='o')


    axx.set_xlabel('Mutual Information Y*Z (Bits)')
    axx.set_ylabel('Information Rates Y*Z (Bits/Min)')
    axx.set_title('Mutual Information vs Information Rate')

    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    c_m = cm.rainbow
    s_m = cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    figx.colorbar(s_m, label='Activator Lambda (minutes)', orientation='vertical')
    figx.savefig('figures/figure4.pdf')

if __name__ == '__main__':
    main()