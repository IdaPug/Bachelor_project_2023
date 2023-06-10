# Import lots thing
import numpy as np                 
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import os

from cil.optimisation.algorithms import GD, FISTA, PDHG, CGLS, SIRT
from cil.optimisation.functions import LeastSquares
from cil.io import TIFFWriter, TIFFStackReader
from cil.framework import AcquisitionGeometry
from cil.processors import CentreOfRotationCorrector, Slicer, \
    Binner, Masker, MaskGenerator, TransmissionAbsorptionConverter, RingRemover

from cil.plugins.astra.processors import FBP
from cil.plugins.astra.operators import ProjectionOperator
from cil.utilities.display import show2D, show_geometry

from mpl_toolkits.axes_grid1 import AxesGrid
#Import Total Variation from the regularisation toolkit plugin
from cil.plugins.ccpi_regularisation.functions import FGP_TV

import astra
astra.astra.set_gpu_index(0)

# Geometry parameters from geometry.m 
#detector parameters
ndet=256                                 # Number of detector pixels, excluding gaps
nElem=2                                  # Number of detector modules
pixel_size=0.077                         # Pixels' size (Pitch)
#Sep=pixel_size*3;
Sep=0.153                  ;               # Pixels' gap length (cm)
det_space=(ndet)*pixel_size+Sep          # Size of detector in cm (pixels*pixel_size), including the gap
#model='fan';                              # Beam geometry (par-fan-lshape-lshape1)

# Geometry of reference curves
#detector parameters
ndet=640;                                  #Number of detector pixels, excluding gaps
nElem=5;                                  # Number of detector modules
pixel_size=0.08025;                          # Pixels' size (Pitch)
Sep=0.386;#pixel_size*3;               # Pixels' gap length (cm)
det_space=ndet*pixel_size+Sep*(nElem-1);  # Size of detector in cm (pixels*pixel_size), including the gap
model='fan';                              # Beam geometry (par-fan-lshape-lshape1)

#acquisition parameters
range_angle=360;                          # Angular span of projections
nproj=2520;                                 # Number of projections
SDD=114.8;#164.7;                                # Source-Detector distance
sourceCentShift=0;#-0.82;#0;                        # Vertical source shift from perfect placement
detectCentShift=0;#25;#0;                        # Vertical detector shift from perfect placement
SAD=23.5;#84.45;                                # Source-AxisOfRotation distance
#vol=true;                                                         
acc_proj = 25;                            


rot_axis_x = 0*pixel_size  # x-position af rotationsaksen
rot_axis_y = 0.0              # y-position af rotationsaksen

# Set up geometry
# Create geometry using CIL
ag = AcquisitionGeometry.create_Cone2D(
    source_position=[sourceCentShift, -SAD],
    detector_position=[detectCentShift, SDD-SAD],
    rotation_axis_position=[rot_axis_x, rot_axis_y])
ag.set_angles(angles=((np.linspace(range_angle,0,nproj, endpoint=False))), angle_unit='degree')
ag.set_panel(num_pixels=[ndet, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels([ 'channel','angle', 'horizontal'])

# Which sinogram to use 
pix_gap=3

# load in h5 filed sinogram
f = h5py.File('../GT_ny/detector_pross/sinogram_GT_'+str(pix_gap)+'_normalFF_right.h5','r')
dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name
   
# Find width of sinogram
num_pix=np.shape(dset)[2]

# Fix geometry fi fit width
ag.set_panel(num_pixels=[num_pix, 1], pixel_size=[pixel_size, pixel_size])

# FIll in data
data = ag.allocate()
data.fill((dset))

# First reorder data to fit with astra
data.reorder('astra')

# Setup RingRemover
wname = "db25"
decNum = 4
sigma = 1.5

# Run RingRemover
data_after_ring_remover = RingRemover(decNum,wname,sigma)(data)


data_sino_ring=data_after_ring_remover.as_array()

cha=64
plt.imshow(data_sino_ring[cha,:,:].T)
plt.title('Sinogram of Ground Truth after Ring_Remover')
plt.colorbar()
plt.clim(0,2.5)
plt.xlabel('Angles')
plt.ylabel('Horizontal')
plt.savefig('Ring_removed.png')
plt.clf()

# Save the ring_removede sinogram
with h5py.File('GT_ringremoved_sino_normalFF.h5', 'w') as hf:
    hf.create_dataset("data",  data=data_sino_ring)

# Set up a fast TV recon to see if the ring remover have an effect
# Create Projection Operator
ig = ag.get_ImageGeometry()
A = ProjectionOperator(ig, ag, 'gpu')

# Try TV solved with FISTA
# Number of iterations
# just do 50 for faster reconstruction
num_itr=200
# ALpha values (should be very low..)
alpha=0.03 # same as GT
f1 = LeastSquares(A, data) # recon on original data
GTV = alpha*FGP_TV(device='gpu') 
myFISTATV = FISTA(f=f1, g=GTV, initial=ig.allocate(0),max_iteration=num_itr, 
update_objective_interval = 10)
#Run it
myFISTATV.run(num_itr,verbose=1)
tv_10= myFISTATV.solution
recon_data=tv_10.as_array()

#show2D(myFISTATV.solution,slice_list=("channel",64),title='FISTA-TV of ground truth without ring removed')
cha=64
plt.imshow(recon_data[cha,:,:],cmap='Greys_r',origin='lower')
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.clim(0,0.9)
plt.title('Ground Truth reconstruction without RingRemover \n channel: '+str(cha))
plt.xlabel('x horizontal')
plt.ylabel('y horizontal')
plt.savefig('GT_NO_ringremoved.png')
plt.clf()


lol=lol[4]
# Also make a profil plot
y=200
plt.plot(recon_data[cha,y,:])
plt.title('Ground Truth reconstruction at y-slice='+str(y)+' \n without RingRemover. channel: '+str(cha))
plt.xlabel('x horizontal')
plt.ylabel('Reconstructed values')
plt.savefig('No_ring_progile.png')
plt.clf()



# Do it all for ring removed data
# Set up a fast TV recon to see if the ring remover have an effect
# Create Projection Operator
ig = ag.get_ImageGeometry()
A = ProjectionOperator(ig, ag, 'gpu')

# Try TV solved with FISTA
# Number of iterations
# just do 50 for faster reconstruction
num_itr=200
# ALpha values (should be very low..)
alpha=0.03 # same as GT
f1 = LeastSquares(A, data_after_ring_remover) # recon on original data
GTV = alpha*FGP_TV(device='gpu') 
myFISTATV = FISTA(f=f1, g=GTV, initial=ig.allocate(0),max_iteration=num_itr, 
update_objective_interval = 10)
#Run it
myFISTATV.run(num_itr,verbose=1)
tv_10= myFISTATV.solution
recon_data=tv_10.as_array()

cha=64
plt.imshow(recon_data[cha,:,:],cmap='Greys_r',origin='lower')
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.clim(0,0.9)
plt.title('Ground Truth reconstruction using RingRemover \n channel: '+str(cha))
plt.xlabel('x horizontal')
plt.ylabel('y horizontal')
plt.savefig('recon_show_alpha_0.03.png')
plt.clf()

# Also make a profil plot
y=200
plt.plot(recon_data[cha,y,:])
plt.title('Ground Truth reconstruction at y-slice='+str(y)+' \n using RingRemover. channel: '+str(cha))
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.xlabel('x horizontal')
plt.ylabel('Reconstructed values')
plt.savefig('profile_plot_alpha_0.03.png')
plt.clf()


# Save ring removed recon 
# Write reconstructed data into tiff files for later use
writer = TIFFWriter(data=tv_10, file_name='Ring_remove_GT_recon/channel', counter_offset=0)
writer.write()