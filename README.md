# MarA-Halflife-2018
The accompanying code and data necessary to reproduce the figures from ("Active degradation of MarA controls coordination of its downstream targets")[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006634] by Nicholas A. Rossi, Thierry Mora, Aleksandra M. Walczak and Mary J. Dunlop



**Figure 1**: This figure contains: simulation, analytical solutions and illustrations. All analytical solutions and simulation results can be achieved by running Figure1_driver.py within the root layer of the codebase. This generates “static.pdf” as well as “ clouds.pdf” with the figures subfolder. These two figures were combined within adobe illustrator with a few illustrations. The growing microcolonies are taken from filmstrips of movies of wild-type E. coli containg the plasmid pVAI (see manuscript). Custom coloring was applied for visual presence in photoshop. Subfigure D is saved from Figure1_driver.py as clouds.pdf 

**Figure 2**: To generate this figure, first run “crispr_driver.py” : this generates all experimental figures as well as the 2A  simulation traces. This script generates the following figure components saved within the figures subfolder : crisp_ts.pdf, crisp_2dhist.pdf, crisp_hist.pdf as well as crispr_simulation_traces.pdf. Next, you can run analytical_driver.py which generates all the analytical solutions to variance over time for Gene X as well as the mutual information over time between gene Y and Z.  The second row of “analytical_solutions.pdf” can be combined with the previous figure components in adobe illustrator. Figure 2D was created exclusively in Illustrator. For the microcolony snapshots, use the raw tiffs contained within the “raw_data” folder. Crispr_driver calls “image_finctions.py” which generates the snapshot images. It saves them independently as “WildType.png” and “CRISPRi.png” before combining them as “combined_images.png”

**Figure 3**: This figure is generated by running “modified_marA_driver.py”.  This calls main which generates all of the experimental figures, and traces which generates the simulation examples. This script generates modified_marA_simulation_traces.pdf as well as modified_marA-TS.pdf. As in figure 2, analytical_driver.py generates the analytic solutions – here the first row shows the analytical solutions to this part. The schematic 3D was created in illustrator. 

**Figure 4**: This figure is generated by running figur4_driver.py. It saves the output as figure4.pdf within the figures folder


