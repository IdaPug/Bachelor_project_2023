import numpy as np
import matplotlib.pyplot as plt
import h5py

from cil.io import TIFFWriter, TIFFStackReader

sino_rep = [1,2,3,4,5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280] 
projs=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]



for i in range(np.shape(projs)[0]):
    
    print('Plotting for projection='+str(projs[i]))
    # Import tiff files for the recon you want to look at
    #reader = TIFFStackReader(file_name = 'Bachelor/aqui_map_recons/comp_varied_alpha_proj_'+str(projs[i])+'_reps_'+str(sino_rep[i]))
    reader = TIFFStackReader(file_name = 'diag_mapping_recons/recon_proj_'+str(projs[i]))
    recon_data = reader.read()

     # plot the data
    cha=64
    plt.imshow(recon_data[cha,:,:],cmap='Greys_r',origin='lower')
    plt.clim(0,0.6)
    cbar = plt.colorbar()
    cbar.set_label('Attenuation [1/cm]')
    plt.title('Recon '+str(projs[i])+' projections and '+str(sino_rep[i])+' repitetions \n channel: '+str(cha))
    plt.xlabel('x horizontal')
    plt.ylabel('y horizontal')
    plt.savefig('diag_recons_look/proj'+str(projs[i])+'.png')
    plt.clf()



