import numpy as np
import matplotlib.pyplot as plt
import h5py

# Projections you want in the plot
proj=[9,35,70,210,2520]

for i in proj:
   # load in GT sinogram
    f = h5py.File('object_values_'+str(i)+'.h5','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name

    # plot the objective value curve
    plt.semilogy(dset/i,label='Projection: '+str(i))

plt.title('Objective value for FISTA-TV reconstruction for different projections \n for Ground Truth data')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('log10(objective value) \n normalised ')
plt.savefig('Obejctive_values_for_different_prof.png')

