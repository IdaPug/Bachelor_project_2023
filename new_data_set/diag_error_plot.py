import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import matplotlib.colors as mcolors

# Load in error value
f = h5py.File('Diag_error_samlet.h5','r')
error = np.array(f["data"][:])

# projections
projs=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]

num_plot=np.shape(projs)[0]


# Make a plot
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)
shown_data=error[1,:]
extent=(error[0,:])
extent = extent.astype('int')

# Also find smallest error:
# Find the best alpha for the projection
idx_min = np.argmin(shown_data)
best_proj=extent[idx_min]
best_error=shown_data[idx_min]

print('Smallest error is at '+str(best_proj)+' projections and it is: '+'%.4f' % best_error)

x_ticks=np.linspace(0,num_plot-1,num_plot)
flip_ticks=np.flip(x_ticks)

plt.plot(np.flip(shown_data),linewidth=3,color='k')
plt.plot(flip_ticks[idx_min],shown_data[idx_min],label='Smallest RMSE',marker='o',markersize=15,color='xkcd:crimson')
plt.legend(fontsize=20)
plt.title('RMSE for reconstruction using constant photon count', fontsize=30)
plt.xlabel('Number of projections',fontsize=20)
plt.ylabel('RMSE',fontsize=20)
plt.xticks(ticks=x_ticks,labels=np.flip(extent),rotation = 45,fontsize=12)
plt.yticks(rotation=45,fontsize=15)
plt.savefig('Diag_error.png')

