# Calculating the error based on the energy curves

import numpy as np
import h5py
import math
from error_funcs_GT_curves import*
from cil.io import TIFFWriter, TIFFStackReader
import random


# Define energy channels where you want to look at the curves. Remember to use 0 indexing
idx_first=19
idx_last=79
curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
curve_span=curve_span.astype(int)

path_to_GT='/work3/s204211/new_data_scripts/energy_curve_fom'
# Energies
# Load Ground truth curves in
#GT_curve=np.hstack((Alu_curve.T,PVC_curve.T))
f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
energies=np.array(f["E_calib"]) # The energy levels captured by the detector
energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)

GT=1
if GT==1:
    # First do Ground truth
    reader = TIFFStackReader(file_name = '../../GT_recon/GT_full_proj_alpha0.03')
    recon_data = reader.read()

    recon_name='Ground Truth'

    error_1, labels1 = small_area_mean_error(recon_data,400,path_to_GT,recon_name,idx_first,idx_last) 
    error_2, labels2 = small_area_error_mean(recon_data,400,path_to_GT,recon_name,idx_first,idx_last)
    error_3, labels3 = area_random_points_mean_error(recon_data,100,path_to_GT,2,recon_name,idx_first,idx_last)
    error_4, labels4 = area_random_points_error_mean(recon_data,100,path_to_GT,2,recon_name,idx_first,idx_last)

    # Make 2D map for error
    #error_plotting=np.dstack((error_1,error_2,error_3,error_4))
    
    #for i in range(np.shape(error_plotting)[1]):
    #    shown_data=error_plotting[:,i,:]
    #    print(np.shape(shown_data))
    #    #plt.imshow(error_plotting[:,:,])



    # Plotting for each material
    for i in range(5):
        plt.plot(error_1[curve_span,i], '*', label='small_area_mean_error')
        plt.plot(error_2[curve_span,i], label='small_area_error_mean')
        plt.plot(error_3[curve_span,i], '*' ,label='area_random_points_mean_error')
        plt.plot(error_4[curve_span,i], label='area_random_points_error_mean')
        plt.legend()
        plt.title('Different error measures for '+labels1[i]+' \n in ground Truth reconstruction \n Curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
        #plt.title('Error based on small_area_median_error for Ground Truth reconstruction \n Material is '+labels1i[i]+' \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1) )
        plt.xlabel('Energy [keV]')
        plt.xticks(ticks=np.linspace(0,idx_last-idx_first,5),labels=np.linspace(energy_span[0],energy_span[-1],5))
        plt.ylabel('Error')
        plt.savefig(labels1[i]+'_plots/Error_proj_Ground_Truth.png')
        plt.clf()


# Big loop for all reconstrictions with best alpha
projs=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]

# Allocate memory for an error vector
error_sum=np.zeros((4,np.shape(projs)[0],5))
# <Error meassuse><projections><materials>

# Allocate memory for an error vector
error_sum_small=np.zeros((4,np.shape(projs)[0],5))
# <Error meassuse><projections><materials>

# Allocate memory for an error vector
error_one_cha=np.zeros((4,np.shape(projs)[0],5))
# <Error meassuse><projections><materials>

# Allocate memory for an error vector
error_full=np.zeros((4,np.shape(projs)[0],128,5))
# <Error meassuse><projections><channels><materials>



for j in range(np.shape(projs)[0]):

    proj=projs[j]
    # Load in reconstruction
    reader = TIFFStackReader(file_name = '../../diag_mapping_recons/recon_proj_'+str(proj))
    recon_data = reader.read()

    recon_name='proj_'+str(proj)
    print(recon_name)

    error_1, labels1 = small_area_mean_error(recon_data,400,path_to_GT,recon_name,idx_first,idx_last) 
    error_2, labels2 = small_area_error_mean(recon_data,400,path_to_GT,recon_name,idx_first,idx_last)
    error_3, labels3 = area_random_points_mean_error(recon_data,20,path_to_GT,2,recon_name,idx_first,idx_last)
    error_4, labels4 = area_random_points_error_mean(recon_data,20,path_to_GT,2,recon_name,idx_first,idx_last)

    # Plotting for each material
    for i in range(5):
        plt.plot(error_1[curve_span,i],'*', label='small_area_mean_error')
        plt.plot(error_2[curve_span,i], label='small_area_error_mean')
        plt.plot(error_3[curve_span,i], '*',label='area_random_points_mean_error')
        plt.plot(error_4[curve_span,i], label='area_random_points_error_mean')
        plt.legend()
        plt.title('Different error measures for '+labels1[i]+' \n in '+str(proj)+' projection reconstruction \n Curves between channel '+str(idx_first+1)+' and '+str(idx_last+1)  )
        #plt.title('Error based on small_area_median_error for Ground Truth reconstruction \n Material is '+labels1i[i]+' \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1) )
        plt.xlabel('Energy [keV]')
        plt.xticks(ticks=np.linspace(0,idx_last-idx_first,5),labels=np.linspace(energy_span[0],energy_span[-1],5))
        plt.ylabel('Error')
        plt.savefig(labels1[i]+'_plots/Error_proj_'+str(proj)+'.png')
        plt.clf()

    error_sum[0,j,:]=np.sum(error_1[1:,:],axis=0)
    error_sum[1,j,:]=np.sum(error_2[1:,:],axis=0)
    error_sum[2,j,:]=np.sum(error_3[1:,:],axis=0)
    error_sum[3,j,:]=np.sum(error_3[1:,:],axis=0)

    error_sum_small[0,j,:]=np.sum(np.abs(error_1[curve_span,:]),axis=0)
    error_sum_small[1,j,:]=np.sum(np.abs(error_2[curve_span,:]),axis=0)
    error_sum_small[2,j,:]=np.sum(np.abs(error_3[curve_span,:]),axis=0)
    error_sum_small[3,j,:]=np.sum(np.abs(error_4[curve_span,:]),axis=0)

    error_one_cha[0,j,:]=error_1[30,:]
    error_one_cha[1,j,:]=error_2[30,:]
    error_one_cha[2,j,:]=error_3[30,:]
    error_one_cha[3,j,:]=error_4[30,:]

    error_full[0,j,:,:]=error_1
    error_full[1,j,:,:]=error_2
    error_full[2,j,:,:]=error_3
    error_full[3,j,:,:]=error_4




# Save the error
with h5py.File('final_GT_error/Error_sum.h5', 'w') as hf:
    hf.create_dataset("data",  data=error_sum)
    hf.create_dataset("projections",data=projs)
    hf.create_dataset("materials",data=labels1)

with h5py.File('final_GT_error/Error_full.h5', 'w') as hf:
    hf.create_dataset("data",  data=error_full)
    hf.create_dataset("projections",data=projs)
    hf.create_dataset("materials",data=labels1)

with h5py.File('final_GT_error/Error_one_cha.h5', 'w') as hf:
    hf.create_dataset("data",  data=error_one_cha)
    hf.create_dataset("projections",data=projs)
    hf.create_dataset("materials",data=labels1)

# Save the error
with h5py.File('final_GT_error/Error_sum_small.h5', 'w') as hf:
    hf.create_dataset("data",  data=error_sum_small)
    hf.create_dataset("projections",data=projs)
    hf.create_dataset("materials",data=labels1)





