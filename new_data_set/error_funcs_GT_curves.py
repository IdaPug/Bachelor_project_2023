import numpy as np
import h5py
import math
import numpy.matlib
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def small_area_mean_error(recon_data,num_pixels,path_to_GT,recon_name,idx_first,idx_last):


    # The path to GT should be at folde containing h5 files for the different GT
    
    # Which channels do we look at
    # Define energy channels where you want to look at the curves. Remember to use 0 indexing
    curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
    curve_span=curve_span.astype(int)

    # number of channels in the data. Assume astra ordering
    num_cha=np.shape(recon_data)[0]

   # Import the GT curve and stack them
    f = h5py.File(path_to_GT+'/Material_curves_from_GT_points.h5','r')
    GT_curves=np.array(f["data"]) # The energy levels captured by the detector
    labels = [x.decode() for x in f['Material_labels']]

    # Unpack the curves
    Alu_GT=GT_curves[:,0]
    PVC_GT=GT_curves[:,2]
    Dic_GT=GT_curves[:,1]
    Etha_GT=GT_curves[:,3]
    Water_GT=GT_curves[:,4]


    #GT_curve=np.hstack((Alu_curve.T,PVC_curve.T))
    # Load in energy levels
    f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
    energies=np.array(f["E_calib"]) # The energy levels captured by the detector
    energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)

    # Calculate the size of the area to look at
    side_size=math.floor(math.sqrt(num_pixels))
    half_side_size=side_size//2
    
    # Define the middel of each material
    alu_mid=[235,200]
    pvc_mid=[190,470]
    etha_mid=[360,410]
    water_mid=[220,320]
    dic_mid=[370,245]

    # Take out reconstructed values in the decided area
    alu = recon_data[:,alu_mid[0]-half_side_size:alu_mid[0]+half_side_size,alu_mid[1]-half_side_size:alu_mid[1]+half_side_size]
    dic = recon_data[:,dic_mid[0]-half_side_size:dic_mid[0]+half_side_size,dic_mid[1]-half_side_size:dic_mid[1]+half_side_size]
    etha = recon_data[:,etha_mid[0]-half_side_size:etha_mid[0]+half_side_size,etha_mid[1]-half_side_size:etha_mid[1]+half_side_size]
    H20 = recon_data[:,water_mid[0]-half_side_size:water_mid[0]+half_side_size,water_mid[1]-half_side_size:water_mid[1]+half_side_size]
    pvc = recon_data[:,pvc_mid[0]-half_side_size:pvc_mid[0]+half_side_size,pvc_mid[1]-half_side_size:pvc_mid[1]+half_side_size]


    # Show where the pixels have been taken from
    # Display the image
    cha=30
    plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
    plt.title('Area selected in each material')
    plt.colorbar()
    # Get the current reference
    ax = plt.gca()

    # Create a Rectangle patch
    rect_alu = Rectangle((alu_mid[1]-half_side_size-1,alu_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='r',facecolor='none',label='Alu')
    rect_etha = Rectangle((etha_mid[1]-half_side_size-1,etha_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='c',facecolor='none',label='Ethanol')
    rect_dica = Rectangle((dic_mid[1]-half_side_size-1,dic_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='m',facecolor='none',label='Dichloromethane')
    rect_pvc = Rectangle((pvc_mid[1]-half_side_size-1,pvc_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='g',facecolor='none',label='PVC')
    rect_H2O = Rectangle((water_mid[1]-half_side_size-1,water_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='b',facecolor='none',label='Water')

    # Add the patch to the Axes
    ax.add_patch(rect_alu)  
    ax.add_patch(rect_etha)
    ax.add_patch(rect_dica)
    ax.add_patch(rect_H2O)
    ax.add_patch(rect_pvc)

    plt.legend()
    plt.savefig('small_area_show.png')
    plt.clf()

    # Initelizing space for saving the error
    Alu_error = np.zeros((num_cha,1))
    dic_error = np.zeros((num_cha,1))
    PVC_error = np.zeros((num_cha,1)) 
    etha_error = np.zeros((num_cha,1))
    Water_error = np.zeros((num_cha,1))

    # Initelize space for plotting arrays
    alu_plot=np.zeros((num_cha,1))
    dic_plot=np.zeros((num_cha,1))
    PVC_plot=np.zeros((num_cha,1))
    etha_plot=np.zeros((num_cha,1))
    water_plot=np.zeros((num_cha,1))

    # mean over the area and find the error
    for i in range(np.shape(alu)[0]): 
        for k in labels:
            Alu_error[i]=np.sqrt((np.mean(alu[i,:,:])-Alu_GT[i])**2)   
            dic_error[i]=np.sqrt((np.mean(dic[i,:,:])-Dic_GT[i])**2) 
            PVC_error[i]=np.sqrt((np.mean(pvc[i,:,:])-PVC_GT[i])**2) 
            etha_error[i]=np.sqrt((np.mean(etha[i,:,:])-Etha_GT[i])**2) 
            Water_error[i]=np.sqrt((np.mean(H20[i,:,:])-Water_GT[i])**2)

        alu_plot[i]=np.mean(alu[i,:,:])
        dic_plot[i]=np.mean(dic[i,:,:])
        PVC_plot[i]=np.mean(pvc[i,:,:])
        etha_plot[i]=np.mean(etha[i,:,:])
        water_plot[i]=np.mean(H20[i,:,:])

    extent=np.linspace(energy_span[0],energy_span[-1],8)
    extent = ["%.2f" % v for v in extent]

    # Also plot the energy curves
    plt.plot(alu_plot[curve_span], label='Alu', color='r')
    plt.plot(dic_plot[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_plot[curve_span], label='PVC', color='g')
    plt.plot(etha_plot[curve_span], label='Ethanol', color='c')
    plt.plot(water_plot[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Energy curves for '+recon_name+' sampled from a '+str(side_size)+'x'+str(side_size)+' area \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    #plt.ylim(0,1.6)
    plt.ylabel('Attenuation value')
    plt.savefig('small_area_mean_error/Energy/_'+recon_name+'.png')
    plt.clf()

    # Also plot the error curves
    plt.plot(Alu_error[curve_span], label='Alu', color='r')
    plt.plot(dic_error[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_error[curve_span], label='PVC', color='g')
    plt.plot(etha_error[curve_span], label='Ethanol', color='c')
    plt.plot(Water_error[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Error curves for '+recon_name+' using method of small area mean error \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Error')
    plt.savefig('small_area_mean_error/Error/_'+recon_name+'.png')
    plt.clf()


    # The labels for the ordering of the curves
    labels=['Alu','Dichloromethane','PVC','Ethanol','Water']
    # Return the error
    error_full=np.hstack((Alu_error,dic_error,PVC_error,etha_error,Water_error))

    return error_full, labels

def small_area_error_mean(recon_data,num_pixels,path_to_GT,recon_name,idx_first,idx_last):
    # The path to GT should be at folde containing h5 files for the different GT
    
    # Which channels do we look at
    # Define energy channels where you want to look at the curves. Remember to use 0 indexing
    curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
    curve_span=curve_span.astype(int)

    # number of channels in the data. Assume astra ordering
    num_cha=np.shape(recon_data)[0]

   # Import the GT curve and stack them
    f = h5py.File(path_to_GT+'/Material_curves_from_GT_points.h5','r')
    GT_curves=np.array(f["data"]) # The energy levels captured by the detector
    labels = [x.decode() for x in f['Material_labels']]

    # Unpack the curves
    Alu_GT=GT_curves[:,0]
    PVC_GT=GT_curves[:,2]
    Dic_GT=GT_curves[:,1]
    Etha_GT=GT_curves[:,3]
    Water_GT=GT_curves[:,4]


    #GT_curve=np.hstack((Alu_curve.T,PVC_curve.T))
    # Load in energy levels
    f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
    energies=np.array(f["E_calib"]) # The energy levels captured by the detector
    energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)



    # Calculate the size of the area to look at
    side_size=math.floor(math.sqrt(num_pixels))
    half_side_size=side_size//2
    
    # Define the middel of each material
    alu_mid=[235,200]
    pvc_mid=[190,470]
    etha_mid=[360,410]
    water_mid=[220,320]
    dic_mid=[370,245]

    # Take out reconstructed values in the decided area
    alu = recon_data[:,alu_mid[0]-half_side_size:alu_mid[0]+half_side_size,alu_mid[1]-half_side_size:alu_mid[1]+half_side_size]
    dic = recon_data[:,dic_mid[0]-half_side_size:dic_mid[0]+half_side_size,dic_mid[1]-half_side_size:dic_mid[1]+half_side_size]
    etha = recon_data[:,etha_mid[0]-half_side_size:etha_mid[0]+half_side_size,etha_mid[1]-half_side_size:etha_mid[1]+half_side_size]
    H20 = recon_data[:,water_mid[0]-half_side_size:water_mid[0]+half_side_size,water_mid[1]-half_side_size:water_mid[1]+half_side_size]
    pvc = recon_data[:,pvc_mid[0]-half_side_size:pvc_mid[0]+half_side_size,pvc_mid[1]-half_side_size:pvc_mid[1]+half_side_size]


    # Show where the pixels have been taken from
    # Display the image
    cha=30
    plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
    plt.title('Area selected in each material')
    plt.colorbar()
    # Get the current reference
    ax = plt.gca()

    # Create a Rectangle patch
    rect_alu = Rectangle((alu_mid[1]-half_side_size-1,alu_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='r',facecolor='none',label='Alu')
    rect_etha = Rectangle((etha_mid[1]-half_side_size-1,etha_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='c',facecolor='none',label='Ethanol')
    rect_dica = Rectangle((dic_mid[1]-half_side_size-1,dic_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='m',facecolor='none',label='Dichloromethane')
    rect_pvc = Rectangle((pvc_mid[1]-half_side_size-1,pvc_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='g',facecolor='none',label='PVC')
    rect_H2O = Rectangle((water_mid[1]-half_side_size-1,water_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='b',facecolor='none',label='H2O')

    # Add the patch to the Axes
    ax.add_patch(rect_alu)  
    ax.add_patch(rect_etha)
    ax.add_patch(rect_dica)
    ax.add_patch(rect_H2O)
    ax.add_patch(rect_pvc)

    plt.legend()
    plt.savefig('small_area_show.png')
    plt.clf()


    # Initelizing space for saving the error
    Alu_error = np.zeros((num_cha,1))
    dic_error = np.zeros((num_cha,1))
    PVC_error = np.zeros((num_cha,1)) 
    etha_error = np.zeros((num_cha,1))
    Water_error = np.zeros((num_cha,1))

    # mean over the area and find the error
    for i in range(np.shape(alu)[0]): 
        Alu_error[i]=np.sqrt(np.mean(((alu[i,:,:]-Alu_GT[i])**2)))
        dic_error[i]=np.sqrt(np.mean(((dic[i,:,:]-Dic_GT[i])**2)))
        PVC_error[i]=np.sqrt(np.mean(((pvc[i,:,:]-PVC_GT[i])**2))) 
        etha_error[i]=np.sqrt(np.mean(((etha[i,:,:]-Etha_GT[i])**2))) 
        Water_error[i]=np.sqrt(np.mean(((H20[i,:,:]-Water_GT[i])**2)))
    
    
    extent=np.linspace(energy_span[0],energy_span[-1],8)
    extent = ["%.2f" % v for v in extent]

    # Also plot the error curves
    plt.plot(Alu_error[curve_span], label='Alu', color='r')
    plt.plot(dic_error[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_error[curve_span], label='PVC', color='g')
    plt.plot(etha_error[curve_span], label='Ethanol', color='c')
    plt.plot(Water_error[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Error curves for '+recon_name+' using method of small area error mean \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Error')
    plt.savefig('small_area_error_mean/Error/_'+recon_name+'.png')
    plt.clf()


    # The labels for the ordering of the curves
    labels=['Alu','Dichloromethane','PVC','Ethanol','Water']
    # Return the error
    error_full=np.hstack((Alu_error,dic_error,PVC_error,etha_error,Water_error))

    return error_full, labels

def area_random_points_mean_error(recon_data,num_points,path_to_GT,seed,recon_name,idx_first,idx_last):
    # The path to GT should be at folde containing h5 files for the different GT
    
    # Which channels do we look at
    # Define energy channels where you want to look at the curves. Remember to use 0 indexing
    curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
    curve_span=curve_span.astype(int)

    # number of channels in the data. Assume astra ordering
    num_cha=np.shape(recon_data)[0]

   # Import the GT curve and stack them
    f = h5py.File(path_to_GT+'/Material_curves_from_GT_points.h5','r')
    GT_curves=np.array(f["data"]) # The energy levels captured by the detector
    labels = [x.decode() for x in f['Material_labels']]

    # Unpack the curves
    Alu_GT=GT_curves[:,0]
    PVC_GT=GT_curves[:,2]
    Dic_GT=GT_curves[:,1]
    Etha_GT=GT_curves[:,3]
    Water_GT=GT_curves[:,4]


    #GT_curve=np.hstack((Alu_curve.T,PVC_curve.T))
    # Load in energy levels
    f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
    energies=np.array(f["E_calib"]) # The energy levels captured by the detector
    energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)

  
    # Define an area to find random pixels in
    # The 3 materials in the containers first
    side_size=70
    half_side_size=side_size//2
    
    # Define the middel of material in class:
    etha_mid=[360,410]
    water_mid=[220,320]
    dic_mid=[370,245]

    # Set seed
    random.seed(seed)

    # The pixel coordinat for the points
    h1=[random.randint(dic_mid[0]-half_side_size, dic_mid[0]+half_side_size) for i in range(num_points)]
    h2=[random.randint(dic_mid[1]-half_side_size, dic_mid[1]+half_side_size) for i in range(num_points)]
    w1=[random.randint(water_mid[0]-half_side_size, water_mid[0]+half_side_size) for i in range(num_points)]
    w2=[random.randint(water_mid[1]-half_side_size, water_mid[1]+half_side_size) for i in range(num_points)]
    s1=[random.randint(etha_mid[0]-half_side_size, etha_mid[0]+half_side_size) for i in range(num_points)]
    s2=[random.randint(etha_mid[1]-half_side_size, etha_mid[1]+half_side_size) for i in range(num_points)]
    

    # Then aluminium and pvc
    alu_mid1=[340,130]
    alu_mid2=[110,290]
    sca=[random.randint(0,100)/(100) for i in range(num_points)]
    move1=[random.randint(-60,60)/(10) for i in range(num_points)]
    move2=[random.randint(-60,60)/(10) for i in range(num_points)]
    a1=((alu_mid1[0]+(alu_mid2[0]-alu_mid1[0])*np.array(sca))+move1).astype(int)
    a2=((alu_mid1[1]+(alu_mid2[1]-alu_mid1[1])*np.array(sca))+move2).astype(int)

    pvc_mid1=[100,400]
    pvc_mid2=[290,530]
    sca=[random.randint(0,100)/(100) for i in range(num_points)]
    move1=[random.randint(-90,90)/(10) for i in range(num_points)]
    move2=[random.randint(-90,90)/(10) for i in range(num_points)]
    p1=((pvc_mid1[0]+(pvc_mid2[0]-pvc_mid1[0])*np.array(sca))+move1).astype(int)
    p2=((pvc_mid1[1]+(pvc_mid2[1]-pvc_mid1[1])*np.array(sca))+move2).astype(int)


    # plot the points found
    cha=30 
    plt.plot(p2,p1,'*g',label='PVC')
    plt.plot(a2,a1,'*r',label='Alu')
    plt.plot(s2,s1,'*c',label='Ethanol')
    plt.plot(h2,h1,'*m',label='Dichloromethane')
    plt.plot(w2,w1,'*b',label='Water')
    plt.legend()
    plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
    plt.title('Random selected points in each material')
    plt.colorbar()
    plt.savefig('Where_points.png')
    plt.clf()
    


   # Initelizing space for saving the error
    Alu_error = np.zeros((num_cha,1))
    dic_error = np.zeros((num_cha,1))
    PVC_error = np.zeros((num_cha,1)) 
    etha_error = np.zeros((num_cha,1))
    Water_error = np.zeros((num_cha,1))

    # Initelize space for plotting arrays
    alu_plot=np.zeros((num_cha,1))
    dic_plot=np.zeros((num_cha,1))
    PVC_plot=np.zeros((num_cha,1))
    etha_plot=np.zeros((num_cha,1))
    water_plot=np.zeros((num_cha,1))

    # mean over the area and find the error
    for i in range(np.shape(alu_plot)[0]): 
        Alu_error[i]=np.sqrt((np.mean(recon_data[i,a1,a2])-Alu_GT[i])**2)   
        dic_error[i]=np.sqrt((np.mean(recon_data[i,h1,h2])-Dic_GT[i])**2) 
        PVC_error[i]=np.sqrt((np.mean(recon_data[i,p1,p2])-PVC_GT[i])**2) 
        etha_error[i]=np.sqrt((np.mean(recon_data[i,s1,s2])-Etha_GT[i])**2) 
        Water_error[i]=np.sqrt((np.mean(recon_data[i,w1,w2])-Water_GT[i])**2) 

        alu_plot[i]=np.mean(recon_data[i,a1,a2])
        dic_plot[i]=np.mean(recon_data[i,h1,h2])
        PVC_plot[i]=np.mean(recon_data[i,p1,p2])
        etha_plot[i]=np.mean(recon_data[i,s1,s2])
        water_plot[i]=np.mean(recon_data[i,w1,w2])

    extent=np.linspace(energy_span[0],energy_span[-1],8)
    extent = ["%.2f" % v for v in extent]


    # Also plot the energy curves
    plt.plot(alu_plot[curve_span], label='Alu', color='r')
    plt.plot(dic_plot[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_plot[curve_span], label='PVC', color='g')
    plt.plot(etha_plot[curve_span], label='Ethanol', color='c')
    plt.plot(water_plot[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Energy curves for '+recon_name+' sampled from a '+str(num_points)+' points each material \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    #plt.ylim(0,1.6)
    plt.ylabel('Attenuation value')
    plt.savefig('area_random_points_mean_error/Energy/_'+recon_name+'.png')
    plt.clf()

    # Also plot the error curves
    plt.plot(Alu_error[curve_span], label='Alu', color='r')
    plt.plot(dic_error[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_error[curve_span], label='PVC', color='g')
    plt.plot(etha_error[curve_span], label='Ethanol', color='c')
    plt.plot(Water_error[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Error curves for '+recon_name+' using method of area random points mean error \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Error')
    plt.savefig('area_random_points_mean_error/Error/_'+recon_name+'.png')
    plt.clf()


    # The labels for the ordering of the curves
    labels=['Alu','Dichloromethane','PVC','Ethanol','Water']
    # Return the error
    error_full=np.hstack((Alu_error,dic_error,PVC_error,etha_error,Water_error))

    return error_full, labels

def area_random_points_error_mean(recon_data,num_points,path_to_GT,seed,recon_name,idx_first,idx_last):
    # The path to GT should be at folde containing h5 files for the different GT
    
    # Which channels do we look at
    # Define energy channels where you want to look at the curves. Remember to use 0 indexing
    curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
    curve_span=curve_span.astype(int)

    # number of channels in the data. Assume astra ordering
    num_cha=np.shape(recon_data)[0]

   # Import the GT curve and stack them
    f = h5py.File(path_to_GT+'/Material_curves_from_GT_points.h5','r')
    GT_curves=np.array(f["data"]) # The energy levels captured by the detector
    labels = [x.decode() for x in f['Material_labels']]

    # Unpack the curves
    Alu_GT=GT_curves[:,0]
    PVC_GT=GT_curves[:,2]
    Dic_GT=GT_curves[:,1]
    Etha_GT=GT_curves[:,3]
    Water_GT=GT_curves[:,4]


    #GT_curve=np.hstack((Alu_curve.T,PVC_curve.T))
    # Load in energy levels
    f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
    energies=np.array(f["E_calib"]) # The energy levels captured by the detector
    energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)

  
    # Define an area to find random pixels in
    # The 3 materials in the containers first
    side_size=70
    half_side_size=side_size//2
    
    # Define the middel of material in class:
    etha_mid=[360,410]
    water_mid=[220,320]
    dic_mid=[370,245]

    # Set seed
    random.seed(seed)

    # The pixel coordinat for the points
    h1=[random.randint(dic_mid[0]-half_side_size, dic_mid[0]+half_side_size) for i in range(num_points)]
    h2=[random.randint(dic_mid[1]-half_side_size, dic_mid[1]+half_side_size) for i in range(num_points)]
    w1=[random.randint(water_mid[0]-half_side_size, water_mid[0]+half_side_size) for i in range(num_points)]
    w2=[random.randint(water_mid[1]-half_side_size, water_mid[1]+half_side_size) for i in range(num_points)]
    s1=[random.randint(etha_mid[0]-half_side_size, etha_mid[0]+half_side_size) for i in range(num_points)]
    s2=[random.randint(etha_mid[1]-half_side_size, etha_mid[1]+half_side_size) for i in range(num_points)]
    

    # Then aluminium and pvc
    alu_mid1=[340,130]
    alu_mid2=[110,290]
    sca=[random.randint(0,100)/(100) for i in range(num_points)]
    move1=[random.randint(-60,60)/(10) for i in range(num_points)]
    move2=[random.randint(-60,60)/(10) for i in range(num_points)]
    a1=((alu_mid1[0]+(alu_mid2[0]-alu_mid1[0])*np.array(sca))+move1).astype(int)
    a2=((alu_mid1[1]+(alu_mid2[1]-alu_mid1[1])*np.array(sca))+move2).astype(int)

    pvc_mid1=[100,400]
    pvc_mid2=[290,530]
    sca=[random.randint(0,100)/(100) for i in range(num_points)]
    move1=[random.randint(-90,90)/(10) for i in range(num_points)]
    move2=[random.randint(-90,90)/(10) for i in range(num_points)]
    p1=((pvc_mid1[0]+(pvc_mid2[0]-pvc_mid1[0])*np.array(sca))+move1).astype(int)
    p2=((pvc_mid1[1]+(pvc_mid2[1]-pvc_mid1[1])*np.array(sca))+move2).astype(int)


    # plot the points found
    cha=30 
    plt.plot(p2,p1,'*g',label='PVC')
    plt.plot(a2,a1,'*r',label='Alu')
    plt.plot(s2,s1,'*c',label='Ethanol')
    plt.plot(h2,h1,'*m',label='Dichloromethane')
    plt.plot(w2,w1,'*b',label='Water')
    plt.legend()
    plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
    plt.title('Random selected points in each material')
    plt.colorbar()
    plt.savefig('Where_points.png')
    plt.clf()
    


   # Initelizing space for saving the error
    Alu_error = np.zeros((num_cha,1))
    dic_error = np.zeros((num_cha,1))
    PVC_error = np.zeros((num_cha,1)) 
    etha_error = np.zeros((num_cha,1))
    Water_error = np.zeros((num_cha,1))



    # mean over the area and find the error
    for i in range(np.shape(Alu_error)[0]): 
        Alu_error[i]=np.sqrt(np.mean((recon_data[i,a1,a2]-Alu_GT[i])**2))
        dic_error[i]=np.sqrt(np.mean((recon_data[i,h1,h2]-Dic_GT[i])**2))
        PVC_error[i]=np.sqrt(np.mean((recon_data[i,p1,p2]-PVC_GT[i])**2)) 
        etha_error[i]=np.sqrt(np.mean((recon_data[i,s1,s2]-Etha_GT[i])**2)) 
        Water_error[i]=np.sqrt(np.mean((recon_data[i,w1,w2]-Water_GT[i])**2)) 

 
    extent=np.linspace(energy_span[0],energy_span[-1],8)
    extent = ["%.2f" % v for v in extent]

    # Also plot the error curves
    plt.plot(Alu_error[curve_span], label='Alu', color='r')
    plt.plot(dic_error[curve_span], label='Dichloromethane',color='m')
    plt.plot(PVC_error[curve_span], label='PVC', color='g')
    plt.plot(etha_error[curve_span], label='Ethanol', color='c')
    plt.plot(Water_error[curve_span], label='Water',color='b')
    plt.legend()
    plt.title('Error curves for '+recon_name+' using method of area random points error mean \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Error')
    plt.savefig('area_random_points_error_mean/Error/_'+recon_name+'.png')
    plt.clf()


    # The labels for the ordering of the curves
    labels=['Alu','Dichloromethane','PVC','Ethanol','Water']
    # Return the error
    error_full=np.hstack((Alu_error,dic_error,PVC_error,etha_error,Water_error))

    return error_full, labels

def random_points_label_mean_error(recon_data,num_points,path_to_GT,seed):
    #number of channels in the data. Assume astra ordering
    num_cha=np.shape(recon_data)[0]
    image_size=np.shape(recon_data)[1]

    # load in ground truth
    # Load Ground truth curves in
    f = h5py.File(path_to_GT,'r')
    GT_Alu = np.array(f["Alatt"][:]) #dataset_name is same as hdf5 object name 
    GT_H2O2 = np.array(f["H2O2att"][:]) #dataset_name is same as hdf5 object name 
    GT_PVC = np.array(f["PVCatt"][:]) #dataset_name is same as hdf5 object name 
    GT_Suger = np.array(f["Sugaratt"][:]) #dataset_name is same as hdf5 object name 
    GT_Water = np.array(f["Wateratt"][:]) #dataset_name is same as hdf5 object name 
    energies=np.array(f["En"]) # The energy levels captured by the detector

    # Which channels do we look at
    # Define energy channels where you want to look at the curves. Remember to use 0 indexing
    idx_first=19
    idx_last=79
    curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
    curve_span=curve_span.astype(int)

    # Selecting random points
    # setting seed
    random.seed(seed)

    # Finding random points
    points1=[random.randint(0+20,image_size-1-20) for i in range(num_points)]
    points2=[random.randint(0+20,image_size-1-20) for i in range(num_points)]
    
    # Show point om image
    cha=30 
    plt.plot(points2,points1,'*')
    plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
    plt.colorbar()
    plt.savefig('heyo.png')

    return points1, points1