import numpy as np                 
import matplotlib.pyplot as plt
import h5py

# Do a loop over all the projections to find the best alpha for each projection
# All the projections
projs=[5,10,37,74,185,370]
#projs=[5]
sino_rep=[10,10,10,5,2,1]


# make room the save the best alpha
alpha_good=np.zeros((1,np.shape(projs)[0]))

for i in range(np.shape(projs)[0]):
    proj = projs[i]

    reps=sino_rep[i]
    # Load in 2D error data
    f = h5py.File('2D_map_error_'+str(proj)+'_smallAlpha_good_ny.h5','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 

    print(np.shape(dset))

    #lol=lol[2]

    # Cut data to not have the NAN values for alpha = 0
    # Cut the data to not have the alpha values i first row
    shown_data = dset[1:,:]
    alpha_ax=dset[0,:].reshape(1,20)
    


    # Find the best alpha for the projection
    idx_min = np.argmin(shown_data)
    best_alpha=alpha_ax[0,idx_min]

    print('Best alpha for '+str(proj)+' projections is: '+str(best_alpha))
    rat=proj/best_alpha
    print('The ratio is '+str(rat))

    # Save the best alpha
    alpha_good[0,i]=best_alpha


    # Axis for plot
    extent = [alpha_ax[0,0] , alpha_ax[0,-1] , 190, 200]

    # Show and save error 2D map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(shown_data,extent = extent, origin='lower')
    #plt.clim(0,0.2)
    ax.set_aspect('auto')
    plt.title('RMSE for '+str(proj)+' projections and '+str(reps)+' acquisition repetitions reconstructed with FISTA - TV')
    plt.xlabel('Alpha')
    plt.ylabel('Iterations')
    plt.colorbar()
    plt.show()
    plt.savefig('Error_maps/2D_map_'+str(proj)+'_small.png')
    plt.clf()

    plt.plot(alpha_ax,shown_data,'*r')
    plt.title('RMSE for '+str(proj)+' projections and '+str(reps)+' acquisition repetitions')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    #plt.xticks(alpha_ax[0,:])
    plt.savefig('Error_maps/2D_map_'+str(proj)+'_plotti.png')

# Save the best alphas
with h5py.File('BEST_ALPHA_small.h5', 'w') as hf:
    hf.create_dataset("data",  data=alpha_good)



