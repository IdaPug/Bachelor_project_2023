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
ag.set_panel(num_pixels=[ndet+4, 1], pixel_size=[pixel_size, pixel_size])
ag.set_channels(128)
ag.set_labels([ 'channel','angle', 'horizontal'])

# Projektion to find the sufficient number of iterations for 
proj=[9,35,70,210,2520]

for i in proj:


    # load in GT sinogram. After Ringremover
    f = h5py.File('GT_ringremoved_sino_normalFF.h5','r')
    dset = np.array(f["data"][:]) #dataset_name is same as hdf5 object name 
    #../GT_ny/detector_pross/
   
    # Cut the dataset to downsize number of projections
    num_proj=i
    cut=nproj//num_proj
    # Cut the dataset to fit with number of projections wanted
    dataset=dset[:,::cut,:]

    # Change geometry to fit with new number of projections
    ag.set_angles(angles=((np.linspace(range_angle,0,num_proj, endpoint=False))), angle_unit='degree')


    # Find width of sinogram
    num_pix=np.shape(dataset)[2]

    # Fix geometry fi fit width
    ag.set_panel(num_pixels=[num_pix, 1], pixel_size=[pixel_size, pixel_size])


    # FIll in data
    data = ag.allocate()
    data.fill((dataset))

    # First reorder data to fit with astra
    data.reorder('astra')


    # Create Projection Operator
    ig = ag.get_ImageGeometry()


    A = ProjectionOperator(ig, ag, 'gpu')

    # Try different alpha reconstruction
    # Try TV solved with FISTA
    # Number of iterations
    num_itr=10
    # ALpha values (should be very low..)
    alpha=0.03


    f1 = LeastSquares(A, data)
    GTV = alpha*FGP_TV(device='gpu') 
    myFISTATV = FISTA(f=f1, g=GTV, initial=ig.allocate(0),max_iteration=200, 
    update_objective_interval = 10)
    myFISTATV.run(num_itr,verbose=1)

    tv_10_before_10=myFISTATV.solution
    # Show the rekonstruction
    #show2D(tv_10_before_10,slice_list=("channel",30),title='FISTA-TV Ground truth after 10 iterations \n ring removed')
    sol_plot_first=tv_10_before_10.as_array()
    plt.imshow(sol_plot_first[63,:,:],origin='lower',cmap='gray')
    plt.title(str(i)+' projection after 10 iterations.\n  Channel: 64')
    cbar = plt.colorbar()
    cbar.set_label('Attenuation [1/cm]')
    plt.clim(0,0.9)
    plt.xlabel('Horizontal x')
    plt.ylabel('Horizonral y')
    plt.savefig('iter_pics/'+str(i)+'_GT_recon_iter_10.png')
    plt.clf()

    
    iters=[20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

    for j in iters:
        print('Running iteration '+str(j))

        #Run it
        myFISTATV.run(10,verbose=1)

        tv_10_now= myFISTATV.solution
        sol_plot=tv_10_now.as_array()


        # Show the rekonstruction
        #show2D(tv_10_now,slice_list=("channel",29),title='FISTA-TV Ground truth after '+str(j)+' iterations \n ring removed')
        
        plt.imshow(sol_plot[64,:,:],origin='lower',cmap='gray')
        plt.title(str(i)+' projections after '+str(j)+' iterations. \n  Channel: 64')
        cbar = plt.colorbar()
        cbar.set_label('Attenuation [1/cm]')
        plt.clim(0,0.9)
        plt.xlabel('Horizontal x')
        plt.ylabel('Horizonral y')
        plt.savefig('iter_pics/'+str(i)+'_GT_recon_iter_'+str(j)+'.png')
        plt.clf()

    
        #tv_10_before=tv_10_now
        # Write reconstructed data into tiff files for later use
        #writer = TIFFWriter(data=tv_10, file_name='../GT_recon/GT_full_proj_alpha'+str(j)+'/channel', counter_offset=0)
        #writer.write()


    # Save the objective values
    with h5py.File('iter_exam_objective_values/object_values_'+str(i)+'.h5', 'w') as hf:
        hf.create_dataset("data",  data=myFISTATV.objective)

    #cha=29
    # make profile plot of the reconstruction
    #plt.clf()
    #sirt_arr=tv_10.as_array()
    #plt.plot(sirt_arr[cha,200,:])
    #plt.xlabel('x horizontal')
    #plt.ylabel('reconstructed value')
    #plt.title('Ground true Profil plot image at y-slice=200 and at channel: '+str(cha)+'\n Detector gap of '+str(i)+' pixel pr gap')
    #plt.savefig('GT_recons_stuff/GT_profil_plot_'+str(i)+'_normalFF_right.png')

    # Write reconstructed data into tiff files for later use
    #writer = TIFFWriter(data=tv_10, file_name='../GT_recon/GT_gap'+str(i)+'right_/channel', counter_offset=0)
    #writer.write()
