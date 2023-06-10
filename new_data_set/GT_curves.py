import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Rectangle
import random


from cil.io import TIFFWriter, TIFFStackReader
# Make ground truth energy curves from GT

# Read in grund truth data
reader = TIFFStackReader(file_name = '../../GT_recon/GT_full_proj_alpha0.03')
recon_data = reader.read()

num_cha=np.shape(recon_data)[0]

#Index in which interval we want to find the curves in. Remember 0-indexing
idx_first=19
idx_last=79
curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
curve_span=curve_span.astype(int)


# Load in energy levels
f = h5py.File('../../Energy_GT_curves/att_coefs_PVC_Alu.mat','r')
energies=np.array(f["E_calib"]) # The energy levels captured by the detector
energy_span=np.linspace(energies[idx_first],energies[idx_last],(idx_last-idx_first)+1)

# Extract matarial curve
# Calculate the size of the area to look at
side_size=20
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
plt.title('20x20 area selected in each material in the ground truth')
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.xlabel('Horozontal x')
plt.ylabel('Horizontal y')
# Get the current reference
ax = plt.gca()

# Create a Rectangle patch
rect_alu = Rectangle((alu_mid[1]-half_side_size-1,alu_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='r',facecolor='none',label='Aluminium')
rect_etha = Rectangle((etha_mid[1]-half_side_size-1,etha_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='c',facecolor='none',label='Ethanol')
rect_dica = Rectangle((dic_mid[1]-half_side_size-1,dic_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='m',facecolor='none',label='Dichloromethane')
rect_pvc = Rectangle((pvc_mid[1]-half_side_size-1,pvc_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='g',facecolor='none',label='PVC')
rect_H2O = Rectangle((water_mid[1]-half_side_size-1,water_mid[0]-half_side_size),side_size,side_size,linewidth=1,edgecolor='b',facecolor='none',label='Water')

# Add the patch to the Axes
ax.add_patch(rect_alu)  
ax.add_patch(rect_dica)
ax.add_patch(rect_pvc)
ax.add_patch(rect_etha)
ax.add_patch(rect_H2O)


plt.legend()
plt.savefig('small_area_show_GT.png')
plt.clf()


# Initelize space for plotting arrays
alu_plot=np.zeros((num_cha,1))
dic_plot=np.zeros((num_cha,1))
PVC_plot=np.zeros((num_cha,1))
etha_plot=np.zeros((num_cha,1))
water_plot=np.zeros((num_cha,1))

# mean over the area and find the energy
for i in range(np.shape(alu)[0]): 
    alu_plot[i]=np.mean(alu[i,:,:])
    dic_plot[i]=np.mean(dic[i,:,:])
    PVC_plot[i]=np.mean(pvc[i,:,:])
    etha_plot[i]=np.mean(etha[i,:,:])
    water_plot[i]=np.mean(H20[i,:,:])

# Only 2 decimals in the plotting
extent=np.linspace(energy_span[0],energy_span[-1],8)
extent = ["%.2f" % v for v in extent]

# Also plot the energy curves
plt.plot(alu_plot[curve_span], label='Aluminium', color='r')
plt.plot(dic_plot[curve_span], label='Dichloromethane',color='m')
plt.plot(PVC_plot[curve_span], label='PVC', color='g')
plt.plot(etha_plot[curve_span], label='Ethanol', color='c')
plt.plot(water_plot[curve_span], label='Water',color='b')
plt.legend()
plt.title('Energy curves construction method: Small area  \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
plt.xlabel('Energy [keV]')
plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
plt.ylim(0,1.6)
plt.ylabel('Attenuation value [1/cm]')
plt.savefig('GT_curves_the_ones_'+str(idx_first)+'_'+str(idx_last)+'.png')
plt.clf()

print(alu_plot.shape)



#Save the curve in the full span
#stack the curves
full_curves=np.hstack((alu_plot,PVC_plot,dic_plot,etha_plot,water_plot))
print(full_curves.shape)
with h5py.File('Material_curves_from_GT.h5', 'w') as hf:
    hf.create_dataset("data",  data=full_curves)
    # Also save material labels
    labels=['Aluminium','PVC','Dichloromethane','Ethanol','Water']
    hf.create_dataset("Material_labels",data=labels)
    

num_points=100

# Do it all again but with random points
# Set seed
seed=1
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
plt.plot(a2,a1,'*r',label='Aluminium')
plt.plot(h2,h1,'*m',label='Dichloromethane')
plt.plot(p2,p1,'*g',label='PVC')
plt.plot(s2,s1,'*c',label='Ethanol')
plt.plot(w2,w1,'*b',label='Water')
plt.legend()
plt.imshow(recon_data[cha,:,:],origin='lower',cmap='gray')
plt.title('100 random selected points in each material')
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.savefig('Where_points_GT.png')
plt.clf()


# Find error
# Initelize space for plotting arrays
alu_plot=np.zeros((num_cha,1))
dic_plot=np.zeros((num_cha,1))
PVC_plot=np.zeros((num_cha,1))
etha_plot=np.zeros((num_cha,1))
water_plot=np.zeros((num_cha,1))


# mean over the area and find the energy
for i in range(np.shape(alu)[0]): 
    alu_plot[i]=np.mean(recon_data[i,a1,a2])
    dic_plot[i]=np.mean(recon_data[i,h1,h2])
    PVC_plot[i]=np.mean(recon_data[i,p1,p2])
    etha_plot[i]=np.mean(recon_data[i,s1,s2])
    water_plot[i]=np.mean(recon_data[i,w1,w2])

# Only 2 decimals in the plotting
extent=np.linspace(energy_span[0],energy_span[-1],8)
extent = ["%.2f" % v for v in extent]

# Also plot the energy curves
plt.plot(alu_plot[curve_span], label='Aluminium', color='r')
plt.plot(dic_plot[curve_span], label='Dichloromethane',color='m')
plt.plot(PVC_plot[curve_span], label='PVC', color='g')
plt.plot(etha_plot[curve_span], label='Ethanol', color='c')
plt.plot(water_plot[curve_span], label='Water',color='b')
plt.legend()
plt.title('Energy curves construction method: Random pixels  \n Energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1))
plt.xlabel('Energy [keV]')
plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
plt.ylim(0,1.6)
plt.ylabel('Attenuation value [1/cm]')
plt.savefig('GT_curves_the_ones_'+str(idx_first)+'_'+str(idx_last)+'points_.png')
plt.clf()

print(alu_plot.shape)


#Save the curve in the full span
#stack the curves
full_curves=np.hstack((alu_plot,dic_plot,PVC_plot,etha_plot,water_plot))
print(full_curves.shape)
with h5py.File('Material_curves_from_GT_points.h5', 'w') as hf:
    hf.create_dataset("data",  data=full_curves)
    # Also save material labels
    labels=['Aluminium','Dichloromethane','PVC','Ethanol','Water']
    hf.create_dataset("Material_labels",data=labels)
    