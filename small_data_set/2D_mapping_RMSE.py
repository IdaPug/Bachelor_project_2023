# Import things
import numpy as np                 
import matplotlib.pyplot as plt
import h5py

from cil.framework import AcquisitionGeometry
from cil.io import TIFFWriter, TIFFStackReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import LeastSquares
from cil.plugins.astra.processors import FBP
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.algorithms import GD, FISTA, PDHG 
from cil.utilities.display import show2D, show_geometry


# Define geometry 
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

rot_axis_x = 0*pixel_size  # x-position af rotationsaksen
rot_axis_y = 0.0              # y-position af rotationsaksen

# Create geometry using CIL
ag = AcquisitionGeometry.create_Cone2D(
    source_position=[sourceCentShift, -SAD],
    detector_position=[detectCentShift, SDD-SAD],
    rotation_axis_position=[rot_axis_x, rot_axis_y])
ag.set_angles(angles=((np.linspace(range_angle,0,nproj, endpoint=False))), angle_unit='degree')
# Importen to set 
ag.set_panel(num_pixels=[ndet+2, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels(['channel','angle', 'horizontal'])

# Load Ground Truth in
reader = TIFFStackReader(file_name = 'Ground_truth/TV/best_data_alpha_0.03')
Ground_Truth = reader.read()

cha=29

# The different sinograms we haves
sino_rep = [10,10,10,5,2,1] # We use with 10 repitition three times.
sino_rep = [10]


# Different projektions we want to try
proj_num=[5,10,37,74,185,370]
proj_num = [5]

# Make error sweep for each projection
for i in range(np.shape(proj_num)[0]):

    gen = sino_rep[i]
    # Load in which sinogram used
    f = h5py.File('sinogram_data/interpol_data_rep_'+str(gen)+'.h5','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 

    num_proj=proj_num[i]
    cut=nproj//num_proj
    # Cut the dataset to fit with number of projections wanted
    dataset=dset[:,::cut,:]

    # Change geometry to fit with new number of projections
    ag.set_angles(angles=((np.linspace(range_angle,0,num_proj, endpoint=False))), angle_unit='degree')

    # FIll in data
    data = ag.allocate()
    data.fill(np.squeeze(dataset))

    # Reorder data to fit with astra
    data.reorder('astra')

    # Create Projection Operator for reconstruction
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, 'gpu')
    f1 = LeastSquares(A, data)

    # Iterations and alpha to loop over
    #alpha_span=np.linspace(0.001,0.1,20)
    alpha_span=np.linspace(0.07,0.07,1)
    #alpha_span=np.logspace(-5,0,6)
    inter_span=np.linspace(200,200,1)
    inter_span = inter_span.astype(int)
    

    # Allocate space for the 2D mapping values 
    error_map = np.zeros((np.shape(inter_span)[0]+1,np.shape(alpha_span)[0]))
    

    for j in range(np.shape(inter_span)[0]):
        for k in range(np.shape(alpha_span)[0]):

            

            ig = ag.get_ImageGeometry()
            recon_data=ig.allocate()

            ag2D=ag.get_slice(channel=0)
            ig2D=ag2D.get_ImageGeometry()
            A_2D = ProjectionOperator(ig2D, ag2D, 'gpu')

            for m in range(ig.channels):
                f1 = LeastSquares(A_2D, data.get_slice(channel=m))
                alpha = alpha_span[k]
                TV = alpha*FGP_TV(device='gpu')

                myFISTATV_2D_FGP = FISTA(f=f1, 
                g=TV, 
                initial=ig2D.allocate(0) ,
                max_iteration=200, 
                update_objective_interval = 10)
                myFISTATV_2D_FGP.run(200,verbrose=1)
                recon_data.fill(myFISTATV_2D_FGP.solution,channel=m)


            # Save the reconstruction
            #show2D(myFISTATV.solution,slice_list=("channel",cha), title='FISTA-TV for '+str(num_proj)+' with alpha: '+str(alpha_span[k])+' and iterations: '+str(num_itr))
            #plt.savefig('2D_maps_error/Recons/'+str(num_proj)+'_proj/TV_alpha_'+str(alpha_span[k])+'_iter_'+str(num_itr)+'.png')

            # Calculate the error
            #sol = myFISTATV.solution.as_array()
            sol = recon_data.as_array()
            sol = sol[cha,:,:]

            GT_sol = Ground_Truth[cha,:,:]

            N=ndet+2
            RMSE = np.sqrt((np.sum(np.subtract(GT_sol,sol)**2))/(N**2))
            print(RMSE)
            # Also save the alpah value
            error_map[j,k]=alpha_span[k]
            error_map[j+1,k]=RMSE

    #Save the error
    with h5py.File('2D_maps_error/2D_map_error_'+str(num_proj)+'_smallAlpha_good_ny.h5', 'w') as hf:
        hf.create_dataset("data",  data=error_map)
