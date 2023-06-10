# Import stuff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import random
import h5py
import math
from cil.io import TIFFWriter, TIFFStackReader


# Load in ground truth curve
path_to_GT='/work3/s204211/new_data_scripts/energy_curve_fom'
# Import the GT curve and stack them
f = h5py.File(path_to_GT+'/Material_curves_from_GT_points.h5','r')
GT_curves=np.array(f["data"]) # The energy levels captured by the detector
labels = [x.decode() for x in f['Material_labels']]

print(labels)

# Unpack the curves
Alu_GT=GT_curves[:,0]
PVC_GT=GT_curves[:,1]
Dic_GT=GT_curves[:,2]
Etha_GT=GT_curves[:,3]
Water_GT=GT_curves[:,4]

# Projection we want to plot the curves from:
projs=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]
projs=[2520,420,140,70,21,9]
#projs=[21]
num_projs=np.shape(projs)[0]

# Array to save the curves in 
curves_plot=np.zeros((128,5,2,num_projs))
#<channels><materials><methods><projections>

# Load in energy levels
idx_first=19
idx_last=79
f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
energies=np.array(f["E_calib"]) # The energy levels captured by the detector
energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)
# Only 2 decimals in the plotting
extent=np.linspace(energy_span[0],energy_span[-1],8)
extent = ["%.2f" % v for v in extent]
curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
curve_span=curve_span.astype(int)

# loop over projections to find curves
for j in range(num_projs):

    i=projs[j]
    print('Projection '+str(i))

    # load in recon
    reader = TIFFStackReader(file_name = '../../diag_mapping_recons/recon_proj_'+str(i))
    recon_data = reader.read()

    # Method 1: small area
    # Calculate the size of the area to look at
    side_size=math.floor(math.sqrt(400))
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

    # Initelize space for plotting arrays
    num_cha=np.shape(recon_data)[0]
    alu_plot=np.zeros((num_cha))
    dic_plot=np.zeros((num_cha))
    PVC_plot=np.zeros((num_cha))
    etha_plot=np.zeros((num_cha))
    water_plot=np.zeros((num_cha))

    # Mean to find curves
    for k in range(np.shape(alu)[0]):
        alu_plot[k]=np.mean(alu[k,:,:])
        dic_plot[k]=np.mean(dic[k,:,:])
        PVC_plot[k]=np.mean(pvc[k,:,:])
        etha_plot[k]=np.mean(etha[k,:,:])
        water_plot[k]=np.mean(H20[k,:,:])


    # Save curves
    curves_plot[:,0,0,j]=alu_plot
    curves_plot[:,1,0,j]=dic_plot
    curves_plot[:,2,0,j]=PVC_plot
    curves_plot[:,3,0,j]=etha_plot
    curves_plot[:,4,0,j]=water_plot


    # Method 2: random pixels
    num_points=10
    #The 3 materials in the containers first
    side_size=70
    half_side_size=side_size//2
    
      # Define the middel of material in class:
    etha_mid=[360,410]
    water_mid=[220,320]
    dic_mid=[370,245]

    # Set seed
    random.seed(2)

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




    # Initelize space for plotting arrays
    num_cha=np.shape(recon_data)[0]
    alu_plot2=np.zeros((num_cha))
    dic_plot2=np.zeros((num_cha))
    PVC_plot2=np.zeros((num_cha))
    etha_plot2=np.zeros((num_cha))
    water_plot2=np.zeros((num_cha))

    # Mean to find curves
    for k in range(np.shape(alu)[0]):
        alu_plot2[k]=np.mean(recon_data[k,a1,a2])
        dic_plot2[k]=np.mean(recon_data[k,h1,h2])
        PVC_plot2[k]=np.mean(recon_data[k,p1,p2])
        etha_plot2[k]=np.mean(recon_data[k,s1,s2])
        water_plot2[k]=np.mean(recon_data[k,w1,w2])

    curves_plot[:,0,1,j]=alu_plot2
    curves_plot[:,1,1,j]=dic_plot2
    curves_plot[:,2,1,j]=PVC_plot2
    curves_plot[:,3,1,j]=etha_plot2
    curves_plot[:,4,1,j]=water_plot2

    # mat points
    Mat_color=['r','m','g','c','b']
    mat_points=np.vstack((a1,a2,h1,h2,p1,p2,s1,s2,w1,w2))
    print(num_points)
    # Plot where points
    for r in range(5):
        plt.imshow(recon_data[63,:,:],origin='lower',cmap='gray')
        for h in range(num_points):
            plt.plot(mat_points[(2*r)+1,h],mat_points[(2*r),h],'*')
        plt.title('Random selected points in '+labels[r])
        plt.colorbar()
        plt.savefig('Where_points_'+labels[r]+'_proj'+str(i)+'.png')
        plt.clf()
        


    # Plot all them point curves
    for r in range(5):
        temp=recon_data[curve_span,:,:]
        fig, ax = plt.subplots()
        p1 = ax.plot(temp[:,mat_points[(2*r),:],mat_points[(2*r)+1]],linestyle='dashed')
        p2 = ax.plot(curves_plot[curve_span,r,1,j],label='Average attenuation',color=Mat_color[r],linewidth=3)
        plt.legend()
        plt.title('Material: '+labels[r]+'. Sampling method: Random Pixels \n with '+str(i)+' projections')
        plt.xlabel('Energy [keV]')
        plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
        plt.ylabel('Attenuation value [1/cm]')
        plt.savefig('diskuss_math2_'+str(labels[r])+'_proj'+str(i)+'.png')
        plt.clf()
  



Mat_color=['r','m','g','c','b']
# Plotting time. Method 1
for m in range(5):
    # GT
    plt.plot(GT_curves[curve_span,m],label=labels[m]+' Ground Truth',color=Mat_color[m])
    for j in range(num_projs):
        plt.plot(curves_plot[curve_span,m,0,j],label='Projections: '+str(projs[j]),linestyle='dashed')
    plt.legend()
    plt.title('Material: '+labels[m]+'. Sampling method: Small Area')
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Attenuation value [1/cm]')
    plt.savefig(labels[m]+'_collective_curves_meth1.png')
    plt.clf()

# Plotting time. Method 2
for m in range(5):
    # GT
    plt.plot(GT_curves[curve_span,m],label=labels[m]+' Ground Truth',color=Mat_color[m])
    for j in range(num_projs):
        plt.plot(curves_plot[curve_span,m,1,j],label='Projections: '+str(projs[j]),linestyle='dashed')
    plt.legend()
    plt.title('Material: '+labels[m]+'. Sampling method: Random Pixels')
    plt.xlabel('Energy [keV]')
    plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
    plt.ylabel('Attenuation value [1/cm]')
    plt.savefig(labels[m]+'_collective_curves_meth2.png')
    plt.clf()