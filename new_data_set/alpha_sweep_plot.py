import numpy as np                 
import matplotlib.pyplot as plt
import h5py

# Do a loop over all the projections to find the best alpha for each projection
# All the projections
# The different sinograms we haves
sino_rep = [1,2,3,4,5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280] 
#sino_rep = [2,3,4,5,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280] 


# Different projektions we want to try
projs=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]
#projs=[1260,840,630,504,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]

# make room the save the best alpha
alpha_good=np.zeros((2,np.shape(projs)[0]))

for i in range(np.shape(projs)[0]):
    proj = projs[i]

    reps=sino_rep[i]
    # Load in 2D error data
    f = h5py.File('../alpha_sweep_error/alpha_error_'+str(proj)+'.h5','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 

    #print(np.shape(dset))


    # Cut data to not have the NAN values for alpha = 0
    # Cut the data to not have the alpha values i first row
    shown_data = dset[1,:]
    alpha_ax=dset[0,:]
    
    print(np.shape(alpha_ax))


    # Find the best alpha for the projection
    idx_min = np.argmin(shown_data)
    best_alpha=alpha_ax[idx_min]

    print('Best alpha for '+str(proj)+' projections is: '+str(best_alpha))
    rat=(proj/best_alpha)
    #print('Ration for '+str(proj)+' is '+str(rat))
    print(alpha_ax)

    # Save the best alpha and projection
    alpha_good[0,i]=proj
    alpha_good[1,i]=best_alpha


    # Axis for plot
    #extent = [alpha_ax[0,0] , alpha_ax[0,-1] , 190, 200]


    plt.plot(alpha_ax,shown_data,'*r')
    plt.title('RMSE for '+str(proj)+' projections and '+str(reps)+' acquisition repetitions')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    #plt.xticks(alpha_ax[0,:])
    plt.savefig('../Alpha_error/error_plot_'+str(proj)+'_sameprodukt.png')
    plt.clf()


# Save the best alphas
with h5py.File('BEST_ALPHA_small.h5', 'w') as hf:
    hf.create_dataset("data",  data=alpha_good)


