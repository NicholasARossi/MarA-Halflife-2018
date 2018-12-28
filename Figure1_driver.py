import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set_style('white')

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


        Gene[t,0] = Gene[t - 1,0] + dt * (-Gene[t - 1,0]/tau_x)+np.random.normal(scale=np.sqrt(dt),loc=0.0)
        Gene[t,1] = Gene[t - 1,1] + dt * (g_y*Gene[t - 1,0]-Gene[t - 1,1]/tau_y)+np.random.normal(loc=0.0,scale=np.sqrt(dt))
        Gene[t, 2] = Gene[t - 1,2] + dt * (g_z*Gene[t - 1,0]-Gene[t - 1,2]/tau_z)+np.random.normal(loc=0.0,scale=np.sqrt(dt))

    return Gene


### here are a set of functions
def varx(t, tau_x):
    return (tau_x / 2) * (1 - np.exp(-2 * t / tau_x))


def vary(t, gy, tau_x, tau_y):
    u = (np.exp(-2 * t * (1 / tau_x + 1 / tau_y)) * tau_y) / (2 * (tau_x - tau_y) ** 2 * (tau_x + tau_y))

    a = np.exp(2 * t / tau_y) * (-1 + np.exp(2 * t / tau_x)) * gy ** 2 * tau_x ** 4 * tau_y

    b = np.exp(2 * t / tau_x) * (-1 + np.exp(2 * t / tau_y)) * tau_x * tau_y ** 2

    c = np.exp(2 * t / tau_x) * (-1 + np.exp(2 * t / tau_y)) * tau_y ** 3

    d = np.exp(2 * t / tau_x) * (-1 + np.exp(2 * t / tau_y)) * tau_x ** 2 * tau_y * (-1 + gy ** 2 * tau_y ** 2)

    e = tau_x ** 3 * (np.exp(2 * t / tau_x) * (-1 + np.exp(2 * t / tau_y)))

    f = tau_x ** 3 * (np.exp(2 * t / tau_x) - 4 * np.exp(t * (1 / tau_x + 1 / tau_y)) + 2 * np.exp(
        2 * t * (1 / tau_x + 1 / tau_y)) + np.exp(2 * t / tau_y)) * gy ** 2 * tau_y ** 2

    return u * (a - b + c + d + e - f)
def covyz_func(t, gy, gz, tx, ty, tz):
    u = (tx ** 2 * ty * tz) / (
    2 * (tx ** 2 - ty ** 2) * (ty + tz) * (tx ** 2 - tz ** 2) * (-2 * ty * tz + tx * (ty + tz)))

    xnaut = np.exp(-t * (3 / tx + 2 / ty + 1 / tz)) * gy ** 2 * ty * (tx + tz)

    ax = -2 * np.exp(t * (3 / tx + 1 / ty)) * (-1 + np.exp(t * (1 / ty + 1 / tz))) * ty ** 2 * tz ** 2

    bx = np.exp(t * (1 / tx + 2 / ty + 1 / tz)) * (-1 + np.exp(2 * t / tx)) * tx ** 3 * (ty + tz)

    cx = np.exp(t * (1 / tx + 1 / ty + 1 / tz)) * (
    -4 * np.exp(t / tx) + 3 * np.exp(t * (2 / tx + 1 / ty)) + np.exp(t / ty)) * tx * ty * tz * (ty + tz)

    dx_0 = np.exp(t * (1 / tx + 1 / ty)) * tx ** 2
    dx = 2 * np.exp(t * (1 / tx + 1 / tz)) * ty ** 2 - np.exp(t * (1 / ty + 1 / tz)) * ty ** 2 + 4 * np.exp(
        t * (1 / tx + 1 / tz)) * ty * tz - 2 * np.exp(2 * t / tx) * tz ** 2 + 2 * np.exp(
        t * (1 / tx + 1 / tz)) * tz ** 2 + np.exp(t * (1 / ty + 1 / tz)) * tz ** 2 - np.exp(
        t * (2 / tx + 1 / ty + 1 / tz)) * (ty ** 2 + 4 * ty * tz + tz ** 2)

    ynaut = np.exp(-t * (3 / tx + 1 / ty + 2 / tz)) * gz ** 2 * (tx + ty) * tz
    ey = -2 * np.exp(t * (3 / tx + 1 / tz)) * ty ** 2 * (tx ** 2 - tz ** 2)
    fy_0 = np.exp(t * (1 / ty + 1 / tz))
    fy = 2 * np.exp(2 * t / tx) * tx * (ty + tz) * (-2 * ty * tz + tx * (ty + tz))
    fy_1 = 2 * np.exp(t * (2 / tx + 1 / tz)) * (tx - ty) * (
    np.sinh(t / tx) * (-tx * ty * tz + ty * tz ** 2 + tx ** 2 * (ty + tz)) + np.cosh(t / tx) * tz * (
    ty * tz - tx * (2 * ty + tz)))

    return u * (xnaut * (ax + bx + cx + dx_0 * dx) + ynaut * (ey + fy_0 * (fy + fy_1)))


def cor2info(corr):
    return -.5 * np.log(1 - corr ** 2)



### all below is for the custom color palette
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return RGB_list

def main():
    plt.close('all')
    labels = ['1 min', '15 mins', '60 mins']
    j = 0
    colors = ['teal', 'coral', 'orchid']
    t_end = 100
    t_int = 0
    N = 400
    dt = float(t_end - t_int) / N
    time_vect = np.arange(t_int, t_end, dt)
    gy = 0.1
    gz = 0.1
    tau_x = 30

    tau_y = 30.1 / np.log(2)

    tau_z = 30.2 / np.log(2)


    tau_x_range = [30.1]


    ### figure c analysis:

    sigy = np.sqrt(vary(time_vect, gy, tau_x / np.log(2), tau_y))
    sigz = np.sqrt(vary(time_vect, gz, tau_x / np.log(2), tau_z))
    covyz = covyz_func(time_vect, gy, gz, tau_x / np.log(2), tau_y, tau_z)
    corr = covyz / (sigy * sigz)




    fig_static = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5])

    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])

    axa1 = plt.subplot(gs00[0])
    axa1.fill_between(time_vect, -np.sqrt(varx(time_vect, tau_x / np.log(2))),
                      np.sqrt(varx(time_vect, tau_x / np.log(2))), color='lightgrey')

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])

    axa2 = plt.subplot(gs01[0])
    axa2.set_ylim([-30, 30])
    axa1.set_ylim([-10, 10])

    axa1.set_xlabel('Time (Minutes)')
    axa1.set_ylabel('Protein X \n Gene Expression')

    axa2.set_xlabel('Time (Minutes)')
    axa2.set_ylabel('Protein Y \n Gene Expression')

    axa3 = plt.subplot(gs01[1])

    axa3.set_xlabel('Time (Minutes)')
    axa3.set_ylabel('Protein Z \n Gene Expression')

    axa3.set_ylim([-30, 30])

    Gene = Simplified_simulation(t_int, t_end, N, tau_x / np.log(2), tau_y, tau_z, gy, gz)
    n_cells = 1000
    temp_vals1 = np.zeros((len(time_vect), n_cells))
    temp_vals2 = np.zeros((len(time_vect), n_cells))
    for z in range(n_cells):
        Gene = Simplified_simulation(t_int, t_end, N, tau_x / np.log(2), tau_y, tau_z, gy, gz)
        temp_vals1[:, z] = Gene[:, 1]
        temp_vals2[:, z] = Gene[:, 2]

    axa1.plot(time_vect, Gene[:, 0], linewidth=3, color=colors[0], alpha=0.6)
    axa2.plot(time_vect, Gene[:, 1], linewidth=3, color=colors[1], alpha=0.6)
    axa3.plot(time_vect, Gene[:, 2], linewidth=3, color=colors[2], alpha=0.6)

    axa2.fill_between(time_vect, -sigy, sigy, color='lightgrey')
    axa3.fill_between(time_vect, -sigy, sigy, color='lightgrey')

    gs02 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2])

    axa4 = plt.subplot(gs[2])
    axa4.plot(time_vect,  cor2info(corr), color='darkslategrey', linewidth=3, label=labels[j])

    axa4.set_xlabel('Time (Minutes)')
    axa4.set_ylabel('Information (Bits)')

    gs03 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs02[1])
    plt.tight_layout()


    # fig_clouds, (axa5a, axa5b, axa5c, axa5d, axa5e) = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(2, 12))

    cloudfig,cloudax=plt.subplots(5,1,sharex=True, sharey=True, figsize=(2, 12))

    cloudax[4].scatter(temp_vals1[0, :],temp_vals2[0, :],alpha=0.1,color='darkslategrey')
    cloudax[3].scatter(temp_vals1[100, :],temp_vals2[100, :],alpha=0.1,color='darkslategrey')

    cloudax[2].scatter(temp_vals1[200, :],temp_vals2[200, :],alpha=0.1,color='darkslategrey')

    cloudax[1].scatter(temp_vals1[300, :],temp_vals2[300, :],alpha=0.1,color='darkslategrey')

    cloudax[0].scatter(temp_vals1[399, :],temp_vals2[399, :],alpha=0.1,color='darkslategrey')

    for l in range(5):
        cloudax[l].axis('off')
        cloudax[l].set_xlim([-50,50])
        cloudax[l].set_ylim([-50, 50])

    # fig_static
    fig_static.savefig('figures/static.png',dpi=300)
    cloudfig.savefig('figures/clouds.png',dpi=300)
    plt.close('all')
    num_cells = 100

    # colors=linear_gradient("#FF7F50","#9370db",n=num_cells)
    colors1 = linear_gradient("#FF7F50", "#F16573", n=50)

    colors2 = linear_gradient("#F16573", "#9370db", n=50)
    colors = np.vstack((colors1, colors2))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6,3))
    ax1.set_xlim([-4,4])
    ax2.set_xlim([-4,4])
    ax1.set_ylim([-4,4])
    ax2.set_ylim([-4,4])
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    # ax1.scatter(x,y,alpha=.5)
    for r in range(1000):
        ax1.scatter(x[r],y[r],alpha=0.2,color=np.divide(colors[49+int((x[r]-y[r])/6.0*49)],255))

    mean = [0, 0]
    cov = [[1, .95], [.95, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T

    ranger=max(x)+max(y)-min(x)-min(y)
    floro=min(x)+min(y)

    for r in range(1000):
        ax2.scatter(x[r],y[r],alpha=0.2,color=np.divide(colors[49+int((x[r]-y[r])/5.5*49)],255))
    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax1.axis('off')
    ax2.axis('off')
    fig.savefig('figures/2clouds.png')

if __name__ == "__main__":
    print("Generating Graphs for Figure 1")
    main()