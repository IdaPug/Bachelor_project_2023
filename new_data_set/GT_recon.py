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


# load in GT sinogram. After Ringremover
f = h5py.File('GT_ringremoved_sino_normalFF.h5','r')
dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 
#../GT_ny/detector_pross/
   

# Find width of sinogram
num_pix=np.shape(dset)[2]

# Fix geometry to fit width
ag.set_panel(num_pixels=[num_pix, 1], pixel_size=[pixel_size, pixel_size])


 # FIll in data
data = ag.allocate()
data.fill((dset))

# First reorder data to fit with astra
data.reorder('astra')

# TV recon for Ground Truth
ig = ag.get_ImageGeometry()
recon_data=ig.allocate()

ag2D=ag.get_slice(channel=0)
ig2D=ag2D.get_ImageGeometry()
A_2D = ProjectionOperator(ig2D, ag2D, 'gpu')

for i in range(ig.channels):
    f1 = LeastSquares(A_2D, data.get_slice(channel=i))
    alpha = 0.03
    TV = alpha*FGP_TV(device='gpu')

    myFISTATV_2D_FGP = FISTA(f=f1, 
                g=TV, 
                initial=ig2D.allocate(0) ,
                max_iteration=200, 
                update_objective_interval = 10)
    myFISTATV_2D_FGP.run(200,verbrose=1)
    recon_data.fill(myFISTATV_2D_FGP.solution,channel=i)

tv_10 = recon_data.as_array()

# Plotting
plt.imshow(tv_10[63,:,:],origin='lower',cmap='gray')
plt.title('Ground Truth reconstruction \n  Channel: 64')
cbar = plt.colorbar()
cbar.set_label('Attenuation [1/cm]')
plt.clim(0,0.9)
plt.xlabel('Horizontal x')
plt.ylabel('Horizonral y')
plt.savefig('GT_recon.png')
plt.clf()



cha=63
# make profile plot of the reconstruction
plt.clf()
plt.plot(tv_10[cha,200,:])
plt.xlabel('x horizontal')
plt.ylabel('reconstructed value')
plt.title('Ground true Profil plot image at y-slice=200 and at channel: '+str(cha)+'\n Detector gap of '+str(i)+' pixel pr gap')
plt.savefig('GT_recons_stuff/GT_profil_plot_'+str(i)+'_normalFF_right.png')

# Write reconstructed data into tiff files for later use
writer = TIFFWriter(data=recon_data, file_name='../GT_recon/GT_/channel', counter_offset=0)
writer.write()
