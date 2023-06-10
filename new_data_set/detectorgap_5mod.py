import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def detectorgap_5mod(sino_data,pix_gap,labels):
    #Input: 
        #Sino_data: np.array with the sinogram data
        #pix_gap: Integer, how many pixels is pr. gap
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

    num_pix_module = num_pix//5
    # First transpose the data to fit into the order ['channel','angle','horizontal']
    dset=np.transpose(sino_data,(idx_ch,idx_an,idx_ho))

    #Make an empty slice as a placeholder to be able to dstack the channels
    sino_int=np.zeros((num_an,num_pix+(pix_gap*4)))

    # Do each channel once at a time
    for i in range(num_ch):

        print(i)

        #Split sinogram into 5 pieces and put extra pixel-space between
        sino_1=dset[i,:,:num_pix_module]
        sino_2=dset[i,:,(num_pix_module):(num_pix_module*2)]
        sino_3=dset[i,:,(num_pix_module*2):(num_pix_module*3)]
        sino_4=dset[i,:,(num_pix_module*3):(num_pix_module*4)]
        sino_5=dset[i,:,(num_pix_module*4):]


        # 1 gap
        sino_gap=np.zeros((num_an,pix_gap))

        # Make final shape
        sino_final_shape=np.hstack((sino_1,sino_gap,sino_2,sino_gap,sino_3,sino_gap,sino_4,sino_gap,sino_5)) 
        #print(np.shape(sino_final_shape))
        #plt.imshow(np.squeeze(sino_final_shape[0:20,126:132]).T)
        #plt.colorbar()
        #plt.savefig('Sino_25_zoom_b_int.png') 

        # Interpol at each angle one at a time
        for j in range(num_an):

            # Find positions of already know values at each angle
            y1=np.linspace(0,num_pix_module-1,num_pix_module)
            y2=np.linspace(num_pix_module+pix_gap,num_pix_module*2+pix_gap-1,num_pix_module)
            y3=np.linspace(num_pix_module*2+pix_gap*2,num_pix_module*3+pix_gap*2-1,num_pix_module)
            y4=np.linspace(num_pix_module*3+pix_gap*3,num_pix_module*4+pix_gap*3-1,num_pix_module)
            y5=np.linspace(num_pix_module*4+pix_gap*4,num_pix_module*5+pix_gap*4-1,num_pix_module)

            # finding values
            Y=np.hstack((y1,y2,y3,y4,y5))

            X=np.ones((num_pix+4*pix_gap))*j
           

            values=sino_final_shape[j,Y.astype(int)]
            #print(np.shape(values))

            Y_new=np.linspace(0,num_pix+4*pix_gap-1,num_pix+4*pix_gap)
            #lin_int= griddata(pos, values, (X, Y), method='linear').T 
            lin_int = np.interp(Y_new, Y.astype(int), values)
            
            sino_final_shape[j,:]=lin_int.T
           
        
        sino_int=np.dstack((sino_int,sino_final_shape))
    
    # Remove placeholder slice
    sino_int = sino_int[:,:,1:]
    
    # Reorder to orginal astra order
    #['channel','angle','horizontal']
    sino_int = np.transpose(sino_int,(2,0,1))
    return sino_int, num_pix+pix_gap*4
