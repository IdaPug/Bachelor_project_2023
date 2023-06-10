# Import things
import numpy as np                 
import matplotlib.pyplot as plt
import h5py

from cil.framework import AcquisitionGeometry
from cil.io import TIFFWriter, TIFFStackReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.plugins.astra.processors import FBP
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.algorithms import GD, FISTA, PDHG 
from cil.utilities.display import show2D, show_geometry

from detectorgap_5mod import*
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
ag.set_panel(num_pixels=[ndet+4, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels([ 'channel','angle', 'horizontal'])


# Load Ground Truth in
reader = TIFFStackReader(file_name = '../GT_recon/GT_full_proj_alpha0.03')
Ground_Truth = reader.read()

# channel to look at error on
cha=29

# Which alpha do we use
alpha_how='Best_alpha'


# The different sinograms we haves
sino_rep = [1,2,3,4,5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280] 
sino_rep = [2]



# Different projektions we want to try
proj_num=[2520,1260,840,630,504,420,360,315,280,252,210,180,168,140,126,120,105,90,84,72,70,63,60, 56, 45, 42, 40, 36, 35, 30, 28, 24, 21,20,18,15,14,12,10,9]
proj_num=[1260]

# Load in best alpha values
f = h5py.File('BEST_ALPHA_small.h5','r')
alpha_val = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 


# Make empty matrix for the errors
error_map=np.zeros((2,np.shape(proj_num)[0]))

# Create loop over all different projections 
for i in range(np.shape(proj_num)[0]):
    print(i)
    gen = sino_rep[i]
    # Load in which sinogram used
    f = h5py.File('../bigdataset/Rawdata_folders/processed'+str(gen)+'/sinogram/sinogram.mat','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 
    dset=np.squeeze(dset[1:-1,:,:])

    #print(np.shape(dset))

   

    # do the interpolation
    labels=['angle','horizontal','channel']

    sino_int, num_pix = detectorgap_5mod(dset,3,labels)

    # Fix geometry to fit the new number of pixels
    ag.set_panel(num_pixels=[num_pix, 1], pixel_size=[pixel_size, pixel_size])
    
    # Cut the dataset
    num_proj=proj_num[i]
    cut=nproj//num_proj
    # Cut the dataset to fit with number of projections wanted
    dataset=sino_int[:,::cut,:]

    # Change geometry to fit with new number of projections
    ag.set_angles(angles=((np.linspace(range_angle,0,num_proj, endpoint=False))), angle_unit='degree')

    # FIll in data
    data = ag.allocate()
    data.fill(dataset)

    # Reorder data to fit with astra
    data.reorder('astra')

    # Create Projection Operator for reconstruction
    ig = ag.get_ImageGeometry()
    recon_data=ig.allocate()

    ag2D=ag.get_slice(channel=0)
    ig2D=ag2D.get_ImageGeometry()
    A_2D = ProjectionOperator(ig2D, ag2D, 'gpu')

    # use the best alpha
    idx_proj=alpha_val[0,:]==num_proj
    alpha=alpha_val[1,idx_proj]
    alpha=alpha[0]

    num_itr=200

    # TV recon
    for m in range(ig.channels):
        f1 = LeastSquares(A_2D, data.get_slice(channel=m))
        # Different alpha
        TV = alpha*FGP_TV(device='gpu')

        myFISTATV_2D_FGP = FISTA(f=f1, 
                        g=TV, 
                        initial=ig2D.allocate(0) ,
                        max_iteration=num_itr, 
                        update_objective_interval = 10)
        myFISTATV_2D_FGP.run(num_itr,verbrose=1)
        recon_data.fill(myFISTATV_2D_FGP.solution,channel=m)


    # Find error
    sol = recon_data.as_array()
    sol = sol[cha,:,:]

    GT_sol = Ground_Truth[cha,:,:]

    N=num_pix
    RMSE = np.sqrt((np.sum(np.subtract(GT_sol,sol)**2))/(N**2))

    # Save the error
    error_map[1,i]=RMSE

    # Also save the projection
    error_map[0,i]=num_proj

    # Save the reconstruction
    writer = TIFFWriter(data=recon_data, file_name='../diag_mapping_recons/recon_proj_'+str(num_proj)+'/channel', counter_offset=0)
    writer.write()


    #Save the error
with h5py.File('Diag_error.h5', 'w') as hf:
    hf.create_dataset("data",  data=error_map)



