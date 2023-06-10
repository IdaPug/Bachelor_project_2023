# Imports
import numpy as np                 
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import os

from cil.optimisation.algorithms import GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction

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

from inter_det_gap_line import*

# Geometry parameters from geometry.m 
#detector parameters
ndet=256                                 # Number of detector pixels, excluding gaps
nElem=2                                  # Number of detector modules
pixel_size=0.077                         # Pixels' size (Pitch)
#Sep=pixel_size*3;
Sep=0.153                  ;               # Pixels' gap length (cm)
det_space=(ndet)*pixel_size+Sep          # Size of detector in cm (pixels*pixel_size), including the gap
#model='fan';                              # Beam geometry (par-fan-lshape-lshape1)

# acquisition parameters
range_angle=360                          # Angular span of projections
nproj=370                                 # Number of projections
SDD=115.0                                # Source-Detector distance
sourceCentShift=0                        # Vertical source shift from perfect placement
detectCentShift=0                        # Vertical detector shift from perfect placement
SAD=57.5                                # Source-AxisOfRotation distance
#acc_proj = 5;                            # number of projections at each the begining and the end where the motor was accelerating/decelerating, should be removed 

rot_axis_x = 0  # x-position af rotationsaksen
rot_axis_y = 0             # y-position af rotationsaksen

# Create geometry using CIL
ag = AcquisitionGeometry.create_Cone2D(
    source_position=[sourceCentShift, -SAD],
    detector_position=[detectCentShift, SDD-SAD],
    rotation_axis_position=[rot_axis_x, rot_axis_y])
ag.set_angles(angles=((np.linspace(range_angle,0,nproj, endpoint=False))), angle_unit='degree')
ag.set_panel(num_pixels=[ndet, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels(['angle', 'horizontal', 'channel'])


# Show geometry
show_geometry(ag)
plt.savefig('Geo.png',transparent='True')


# Import sinograms from matlab and load into a numpy array
# Which sinogram to load in
gen = 10 # How many repetitions
f = h5py.File('sinogram_data/sinogram_gen'+str(gen)+'.mat','r')
dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name

# Put data into CIL class
data = ag.allocate()
data.fill(np.squeeze(dset))

# We now try to get at better reconstruction by getting rid of dead/hot pixels
# Make mask for pixels we don't want
mask = MaskGenerator.mean(threshold_factor=1.5, window=5)(data)


# Show the mask
show2D(mask,slice_list=("channel",63))
plt.savefig('mask_25.png') 

# Interpolate the masked data
data_masked = Masker.interpolate(mask=mask, method='nearest', axis='horizontal')(data)

# Do the interpolation of the detector gap using homemade function
sino_int, newpix=inter_det_gap(data_masked.as_array(),2,['angle', 'horizontal', 'channel'])

# Change the panel geometry to fit the new sinogram size
ag.set_panel(num_pixels=[newpix, 1], pixel_size=[pixel_size, pixel_size])
# Change labels to fit with astra
ag.set_labels(['channel','angle','horizontal'])


#Save the masked and interpolated data 
with h5py.File('interpol_data_rep_'+str(gen)+'.h5', 'w') as hf:
   hf.create_dataset("data",  data=sino_int)

# Display final sinogram
s_chal=63 #Choose which energy channel to look at
plt.imshow(np.squeeze(dset[:,:,:,s_chal]).T)
plt.savefig('Final_Sino_'+str(s_chal)+'.png') 


# Try to do FBP
# allocate space for the FBP recon
ag2D = ag.get_slice(channel=0)
ig2D = ag2D.get_ImageGeometry()
ig = ag.get_ImageGeometry()
FBP_recon_3D = ig.allocate()

# FBP reconstruction per channel
for i in range(ig.channels):
    
    FBP_recon_2D = FBP(ig2D, ag2D, 'gpu')(data_masked.get_slice(channel=i))
    FBP_recon_3D.fill(FBP_recon_2D, channel=i)
    
    print("Finish FBP recon for channel {}".format(i), end='\r')
    
print("\nFDK Reconstruction Complete!")

# Write reconstructed data into tiff files for later use
#writer = TIFFWriter(data=FBP_recon_3D, file_name='3D_recon_output/25_data/FBP/data_write', counter_offset=0)
#writer.write()

show2D(FBP_recon_3D,slice_list=("channel",63))
plt.savefig('fbp_full.png')


# Do TV reconstruction with FISTA
# Create Projection Operator
TV_recon_3D = ig.allocate()

A = ProjectionOperator(ig2D, ag2D, device="gpu")


# TV reconstruction per channel
for i in range(ig.channels):
    b = data.get_slice(channel=i)
    # least squares
    f1 = LeastSquares(A, b)

    alpha=0.03
    num_itr=100 
    GTV = alpha*FGP_TV(device='gpu') 
    myFISTATV = FISTA(f=f1, 
    g=GTV, 
    initial=ig2D.allocate(0),
    max_iteration=num_itr, 
    update_objective_interval = 10)
    
    myFISTATV.run(num_itr,verbose=1)
    TV_recon_3D.fill(myFISTATV.solution, channel=i)
    
    print("Finish tv recon for channel {}".format(i), end='\r')
    
print("\n TV Reconstruction Complete!")

# Show solution
show2D(TV_recon_3D,slice_list=("channel",64))
plt.savefig('TV_recon_'+str(alpha)+'.png')

# Write reconstructed data into tiff files for later use
#writer = TIFFWriter(data=tv_sol, file_name='3D_recon_output/25_data/TV/data_write', counter_offset=0)
#writer.write()



