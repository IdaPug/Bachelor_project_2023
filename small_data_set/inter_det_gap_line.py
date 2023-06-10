import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def inter_det_gap(sino_data,pix_gap,labels):
    #Input: 
        #Sino_data: np.array with the sinogram data
        #pix_gap: Integer, how many pixels is the gap
        #labels: list, string with labels describing which order the sinogram data is in
    
    #Output:
        #New sinogram with interpolated gap. IN ASTRA ORDERING: ['channel','angle','horizontal']
        #Interger with new size of pixels 


    # Find out which dimension is which
    idx_ho=labels.index('horizontal')
    idx_ch=labels.index('channel')
    idx_an=labels.index('angle')
    num_pix=np.shape(sino_data)[idx_ho]
    num_ch=np.shape(sino_data)[idx_ch]
    num_an=np.shape(sino_data)[idx_an]
    print(num_pix)
    num_pix_mid=num_pix//2
    # First transpose the data to fit into the order ['channel','angle','horizontal']
    dset=np.transpose(sino_data,(idx_ch,idx_an,idx_ho))

    #Make an empty slice as a placeholder to be able to dstack the channels
    sino_int=np.zeros((num_an,num_pix+pix_gap))

    # Do each channel once at a time
    for i in range(num_ch):

        #Split sinogram into 2 pieces and put extra pixel-space between
        sino_first=dset[i,:,:num_pix_mid]
        sino_second=dset[i,:,num_pix_mid:]
        sino_gap=np.zeros((num_an,pix_gap))
        sino_final_shape=np.hstack((sino_first,sino_gap,sino_second)) 

        #plt.imshow(np.squeeze(sino_final_shape[0:20,126:132]).T)
        #plt.colorbar()
        #plt.savefig('Sino_25_zoom_b_int.png') 

        # Interpol at each angle one at a time
        for j in range(num_an):

            # Find positions of already know values
            x = np.linspace(j,j,1)
            y = np.linspace(0, num_pix_mid-1, num_pix_mid)
            X_first, Y_first = np.meshgrid(x,y)

            x = np.linspace(j,j,1)
            y = np.linspace(num_pix_mid+pix_gap, num_pix+pix_gap-1, num_pix_mid)
            X_second, Y_second = np.meshgrid(x,y)

            positions_first = np.vstack([X_first.ravel(), Y_first.ravel()]).T
            positions_second = np.vstack([X_second.ravel(), Y_second.ravel()]).T
            positions=np.vstack([positions_first,positions_second])
            pos=positions.astype(int)

            #Meshgrid and interpolate
            values=sino_final_shape[pos[:,0],pos[:,1]]
            ny, nx = sino_final_shape.shape[1], sino_final_shape.shape[0]
            X, Y = np.meshgrid(np.linspace(j,j,1), np.arange(0, ny, 1))
            #lin_int= griddata(pos, values, (X, Y), method='linear').T 
            lin_int = np.interp(Y, pos[:,1], values)
            sino_final_shape[j,:]=lin_int.T
           
        
        sino_int=np.dstack((sino_int,sino_final_shape))
    
    # Remove placeholder slice
    sino_int = sino_int[:,:,1:]
    
    # Reorder to orginal astra order
    #['channel','angle','horizontal']
    sino_int = np.transpose(sino_int,(2,0,1))
    return sino_int, num_pix+pix_gap
        
    


    

    
