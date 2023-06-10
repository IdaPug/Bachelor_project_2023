# make and view reconstructions to use for Ground truth. 
# SIRT and TV

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

# acquisition parameters
range_angle=360                          # Angular span of projections
nproj=370                                 # Number of projections
SDD=115.0                                # Source-Detector distance
sourceCentShift=0                        # Vertical source shift from perfect placement
detectCentShift=0                        # Vertical detector shift from perfect placement
SAD=57.5                                # Source-AxisOfRotation distance
#acc_proj = 5;                            # number of projections at each the begining and the end where the motor was accelerating/decelerating, should be removed 

rot_axis_x = 0                # x-position af rotationsaksen
rot_axis_y = 0.0              # y-position af rotationsaksen

# Create geometry using CIL
ag = AcquisitionGeometry.create_Cone2D(
    source_position=[sourceCentShift, -SAD],
    detector_position=[detectCentShift, SDD-SAD],
    rotation_axis_position=[rot_axis_x, rot_axis_y])
ag.set_angles(angles=((np.linspace(range_angle,0,nproj, endpoint=False))), angle_unit='degree')
ag.set_panel(num_pixels=[ndet+2, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels(['channel','angle', 'horizontal'])


# Load in interpolated and masked sinograms
f = h5py.File('../sinogram_data/interpol_data_rep_10.h5','r')
dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 

# FIll in data
data = ag.allocate()
data.fill(np.squeeze(dset))

# First reorder data to fit with astra
data.reorder('astra')

# SIRT Ground Truth reconstruction
# Number of iterations
num_itr=800

ig = ag.get_ImageGeometry()

A = ProjectionOperator(ig, ag, 'gpu')

sirt = SIRT(initial = ig.allocate(0),operator = A, data = data,
                max_iteration = num_itr, lower=0, update_objective_interval = 10)

sirt.run(num_itr,verbrose=1)
sirt_20=sirt.solution

show2D(sirt_20,slice_list=("channel",29),title='SIRT with '+str(num_itr)+' iterations')
plt.savefig('SIRT/'+str(num_itr)+'_iter_hej.png')

# make profile plot of the reconstruction
plt.clf()
sirt_arr=sirt_20.as_array()
plt.plot(sirt_arr[63,:,150])
plt.xlabel('y horizontal')
plt.ylabel('reconstructed value')
plt.title('Profil of SIRT reconstructed image at x-slice=150 and interations:' +str(num_itr))
plt.savefig('SIRT/profil_plot/profil_'+str(num_itr)+'_iter.png')


# Write reconstructed data into tiff files for later use
writer = TIFFWriter(data=sirt_20, file_name='SIRT/800_data/channel', counter_offset=0)
writer.write()




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
                max_iteration=100, 
                update_objective_interval = 10)
    myFISTATV_2D_FGP.run(200,verbrose=1)
    recon_data.fill(myFISTATV_2D_FGP.solution,channel=i)

tv_10 = recon_data

show2D(tv_10,slice_list=("channel",63),title='FISTA-TV with '+str(num_itr)+'iterations and alpha: '+str(alpha))
plt.savefig('TV/'+str(num_itr)+'_iter_alpha_'+str(alpha)+'_new.png')

# make profile plot of the reconstruction
plt.clf()
arr=tv_10.as_array()
plt.plot(arr[63,:,150])
plt.xlabel('y horizontal')
plt.ylabel('reconstructed value')
plt.title('Profil of TV reconstructed image at x-slice=150 and alpha: '+str(alpha))
plt.savefig('TV/profil_plot/profil_'+str(num_itr)+'_iter_new.png')

# Write reconstructed data into tiff files for later use
writer = TIFFWriter(data=tv_10, file_name='TV/best_data_alpha_0.03_better/channel', counter_offset=0)
writer.write()
