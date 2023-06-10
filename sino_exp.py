# Small script to make exampels of sinogram from imaga and exampel of ringremover

# CIL core components needed
from cil.framework import ImageGeometry, AcquisitionGeometry, BlockDataContainer

# CIL optimisation algorithms and linear operators
from cil.optimisation.algorithms import CGLS
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator, FiniteDifferenceOperator

# CIL example synthetic test image
from cil.utilities.dataexample import SHAPES

# CIL display tools
from cil.utilities.display import show2D, show_geometry

# Forward/backprojector from CIL ASTRA plugin
from cil.plugins.astra import ProjectionOperator

# For shepp-logan test image in CIL tomophantom plugin
from cil.plugins import TomoPhantom as cilTomoPhantom

# Ring remove
from cil.processors import RingRemover

# Third-party imports
import numpy as np    
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import cv2




# Make a Phantom to make an example sinogram. Just a square with stuff in it
n = 256

# Function to generate an random image
def generate_gray_image_with_shapes(K):
    # Create a new blank image with size K
    img = Image.new("RGB", (K, K), color="white")
    draw = ImageDraw.Draw(img)

    # Randomly generate size and position of a square
    square_size = random.randint(K // 10, K // 2)
    square_x = random.randint(0, K - square_size)
    square_y = random.randint(0, K - square_size)

    # Randomly generate size and position of a circle
    circle_size = random.randint(K // 10, K // 2)
    circle_x = random.randint(0, K - circle_size)
    circle_y = random.randint(0, K - circle_size)

   

    # Check if the square and circle intersect or touch each other
    while (square_x < circle_x + circle_size and square_x + square_size > circle_x and
           square_y < circle_y + circle_size and square_y + square_size > circle_y):
        # If they do, generate new random positions for the square and circle
        square_size = random.randint(K // 10, K // 2)
        square_x = random.randint(0, K - square_size)
        square_y = random.randint(0, K - square_size)
        circle_size = random.randint(K // 10, K // 2)
        circle_x = random.randint(0, K - circle_size)
        circle_y = random.randint(0, K - circle_size)

    # Draw the square and circle on the image
    draw.rectangle((square_x, square_y, square_x + square_size, square_y + square_size), fill="blue")
    draw.ellipse((circle_x, circle_y, circle_x + circle_size, circle_y + circle_size), fill="red")

    
    # Convert the PIL image to a grayscale NumPy array
    gray_img = np.dot(np.array(img, dtype=np.float32), [0.2989, 0.5870, 0.1140])
    
    # Normalize the grayscale values between 0 and 1
    gray_img = gray_img / 255.
    gray_img = 1 - gray_img
    
    return gray_img

im_ex = generate_gray_image_with_shapes(n)

plt.imshow(im_ex,cmap='Greys')
plt.savefig('exp_imaga.png')
plt.clf()

# Set up image geometry
ig = ImageGeometry(voxel_num_x=n, 
                   voxel_num_y=n, 
                   voxel_size_x=2/n, 
                   voxel_size_y=2/n)
print(ig)

# Then we set up the Acquisition Geometry
num_angles = 360
ag = AcquisitionGeometry.create_Parallel2D()  \
                   .set_angles(np.linspace(0, 360, num_angles, endpoint=False))  \
                   .set_panel(n, 2/n)
print(ag)

# Make datacontainer and fill with simulated image
sim_img = ig.allocate()
sim_img.fill(im_ex)


# Simulate sinogram with projections operater
device = "gpu"
A = ProjectionOperator(ig, ag, device)

# Plotting
sinogram = A.direct(sim_img)
plots = [sim_img, sinogram]
titles = ["Ground truth", "sinogram"]
show2D(plots, titles)
plt.title('Sinogram for simulated image data')
plt.savefig('sino_AAAAAA_2.png')

#print(np.shape(sinogram))


# Add noise that will appear as rings in the recon
# Function to add noise to an array
def add_noise(array, noise_type='gaussian', scale=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    noisy_array = array.copy()  # Create a copy of the original array

    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0, scale=scale, size=array.shape)
        noisy_array += noise
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-scale, high=scale, size=array.shape)
        noisy_array += noise
    elif noise_type == 'salt-and-pepper':
        mask = np.random.choice([0, 1, 2], size=array.shape, p=[0.05, 0.05, 0.9])
        noise = np.random.uniform(low=-scale, high=scale, size=array.shape)
        noisy_array = np.where(mask == 0, 0, np.where(mask == 1, 255, noisy_array + noise))
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Supported types are 'gaussian', 'uniform', 'salt-and-pepper'.")

    return noisy_array

# Add noise to all angels
sinogram_arr=sinogram.as_array()
for j in range(np.shape(sinogram)[0]):
    sinogram_arr[j,:]=add_noise(sinogram_arr[j,:],noise_type='gaussian', scale=0.05, seed=1)


# fill new sinogram into cil datacontainer
sino_noise = ag.allocate()
sino_noise.fill(sinogram_arr)

# Do a ring remover on it
# Setup and run RingRemover Processor
wname = "db25"
decNum = 4
sigma = 1.5

data_after_ring_remover = RingRemover(decNum=4, wname="db25", sigma=1.5)(sino_noise)

# Show together with original sino
plots = [sino_noise, data_after_ring_remover]
titles = ["Before RingRemover", "After RingRemover"]
show2D(plots, titles)
#plt.title('Effect of Ringremover')
plt.savefig('sino_ringremoved_exp.png')

 