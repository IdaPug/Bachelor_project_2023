# Import lots of thing
import numpy as np                 
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import os
from detectorgap_5mod import*

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

#from inter_det_gap_line import*

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
ag.set_labels(['angle', 'horizontal', 'channel'])

# Show geo
show_geometry(ag)
plt.savefig('geo.png',transparent='True')

# interpolate with different gap sizes
for i in range(1,6):
    print(i)
    # Import sinogram 
    f = h5py.File('../GT_ny/sinogram/sinogram_good.mat','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name

    dset=np.squeeze(dset)
    print(np.shape(dset))


    plt.imshow(dset[:,:,64].T)
    plt.colorbar()
    plt.title('Singram for Ground Truth with normal flat field')
    #plt.savefig('GT_sino_normalFF.png')
    plt.clf()


    # Put data into CIL class
    data = ag.allocate()
    data.fill(np.squeeze(dset[1:-1,:,:]))

 

    # Reorder to fit with astra
    data.reorder('astra')

    dset=data.as_array()

    labels=['channel','angle','horizontal']

    sino_int, num_pix = detectorgap_5mod(dset,i,labels)
    print(num_pix)

    #labels_ff=[5,6,7,8,9,10,11,12,13,14]
    # Try to get the flat field from the sinogram backsgroun
    #for i in range(10):
    #    plt.plot(np.squeeze(dset[64,:,5+i]),label=labels_ff[i])
    #plt.legend(bbox_to_anchor=(1.04,1),loc="upper left")
    #plt.title('Background pixels along all the angles from the sinogram \n in selected pizels in the horizontal axis')
    #plt.savefig('Roling_FF.png',bbox_inches="tight")

        

    with h5py.File('../GT_ny/detector_pross/sinogram_GT_'+str(i)+'_normalFF_right.h5', 'w') as hf:
        hf.create_dataset("data",  data=sino_int)

    # showing of the sinogram
    plt.imshow(sino_int[64,:,:].T)
    plt.colorbar()
    plt.title('Singram with detector gap '+str(i)+'pr. gap')
    plt.savefig('GT_sino_'+str(i)+'pixgap_normalFF_right.png')
    plt.clf()



