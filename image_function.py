import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt



def make_rgb(image):
    return (image * 255).round().astype(np.uint8)

def create_combined_image(bases,pscale,cscale,yscale,rscale):
    fig, ax = plt.subplots(2, 1, figsize=(4, 10))
    strains=['WildType','CRISPRi']
    for z in [0,1]:
        base=bases[z]
        phase_names, c_names, y_names, r_names=base+'c1.tif',base+'c2.tif',base+'c3.tif',base+'c4.tif'


        cim = np.array(mpimg.imread(c_names))
        pim = np.array(mpimg.imread(phase_names))
        yim = np.array(mpimg.imread(y_names))
        rim = np.array(mpimg.imread(r_names))

        # rescaling values


        cim[cim > cscale[1]] = cscale[1]
        cim[cim < cscale[0]] = cscale[0]

        yim[yim > yscale[1]] = yscale[1]
        yim[yim < yscale[0]] = yscale[0]

        rim[rim > rscale[1]] = rscale[1]
        rim[rim < rscale[0]] = rscale[0]

        pim[pim > pscale[1]] = pscale[1]
        pim[pim < pscale[0]] = pscale[0]

        cim = (cim - cscale[0]) / (cscale[1] - cscale[0])
        pim = (pim - pscale[0]) / (pscale[1] - pscale[0])
        yim = (yim - yscale[0]) / (yscale[1] - yscale[0])
        rim = (rim - rscale[0]) / (rscale[1] - rscale[0])
        yim=yim/np.max(yim)
        rim=rim/np.max(rim)
        # combined_array=np.zeros
        cfp_img = []
        yfp_img = []
        phase_img = []

        combined_img = []
        for i in range(len(cim[:])):
            c_row = []
            p_row = []
            # y_row = []
            combined_row = []
            for j in range(len(cim[1])):
                c_temp_val = [0, cim[i][j], cim[i][j]]
                p_temp_val = [pim[i][j], pim[i][j], pim[i][j]]
                y_temp_val = [yim[i][j], yim[i][j], 0]


                # combined_temp_val = [rim[i][j]+ pim[i][j],  pim[i][j], pim[i][j]+yim[i][j]]
                combined_temp_val = [pim[i][j]+rim[i][j],  pim[i][j], pim[i][j]+yim[i][j]]

                # combined_temp_val = [pim[i][j]+rim[i][j],  pim[i][j]+.5*yim[i][j]+.5*rim[i][j], pim[i][j]]

                # combined_temp_val = [pim[i][j]+rim[i][j],  pim[i][j]+.5*rim[i][j], pim[i][j]]

                # combined_temp_val = [0 + pim[i][j] , cim[i][j] + pim[i][j],
                #                      cim[i][j] + pim[i][j]]
                c_row += [c_temp_val]
                p_row += [p_temp_val]
                # y_row += [y_temp_val]
                combined_row += [combined_temp_val]
            cfp_img.append(c_row)
            # yfp_img.append(y_row)
            phase_img.append(p_row)
            combined_img.append(combined_row)
        imsave('figures' + '/' + strains[z] + '.png', combined_img)
        # ax[z].imshow(np.asarray(combined_img)[50:450, 150:550])


    im = mpimg.imread('figures/WildType.png')
    ax[1].imshow(im[50:450, 150:550])
    im = mpimg.imread('figures/CRISPRi.png')
    ax[0].imshow(im[85:485, 250:650])
    ax[0].axis('off')
    ax[1].axis('off')
    fig.savefig('figures/combined_images.png',dpi=200)




if __name__ == '__main__':
    pscale = [500, 3000]
    cscale = [2000,16000]

    yscale = [700, 16000]
    rscale = [600, 16000]


    bases=['raw_data/0_none_redux/xy2/nonet90xy2','raw_data/1_14_extracts/14t90xy5']

    # yscale = [700, 3000]
    # rscale = [600, 1200]
    create_combined_image(bases,pscale,cscale,yscale,rscale)