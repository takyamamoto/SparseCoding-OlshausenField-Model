# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm
import scipy.io as sio

np.random.seed(0)

def DoG(img, ksize=(5,5), sigma=1.3, k=1.6):
    # DoG filter as a model of LGN
    g1 = cv2.GaussianBlur(img, ksize, sigma)
    g2 = cv2.GaussianBlur(img, ksize, k*sigma)
    dog = g1 - g2
    return dog
    #return (dog - dog.min())/(dog.max()-dog.min())

def GaussianMask(sizex=16, sizey=16, sigma=4.8):
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    x0 = sizex // 2
    y0 = sizey // 2
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(sigma**2)))
    return mask / np.sum(mask)

# Preprocess of inputs
num_iter = 5000

imgdirpath = "./images_preprocessed/"
imglist = []

# datasets from http://www.rctn.org/bruno/sparsenet/
mat_images = sio.loadmat('datasets/IMAGES.mat')
imgs = mat_images['IMAGES']
mat_images_raw = sio.loadmat('datasets/IMAGES_RAW.mat')
imgs_raw = mat_images_raw['IMAGESr']
    
# Define model
model = network.RaoBallard1999Model()

# Simulation constants
H, W, num_images = imgs.shape
nt_max = 1000 # Maximum number of simulation time
eps = 1e-3 # small value which determines convergence
input_scale = 40 # scale factor of inputs
gmask = GaussianMask() # Gaussian mask
error_list = [] # List to save errors

for iter_ in tqdm(range(num_iter)):
    # Get images randomly
    idx = np.random.randint(0, num_images)
    img = imgs[:, :, idx]
    
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-27)
    beginy = np.random.randint(0, H-17)
    img_clopped = img[beginy:beginy+16, beginx:beginx+26]

    # Clop three inputs
    inputs = np.array([(gmask*img_clopped[:, i*5:i*5+16]).flatten() for i in range(3)])
    inputs = (inputs - np.mean(inputs)) * input_scale
    
    # Reset states
    model.initialize_states(inputs)
    
    # Input an image patch until latent variables are converged 
    for i in range(nt_max):
        # Update r and rh without update weights 
        error, errorh, dr, drh = model(inputs, training=False)
        
        # Compute norm of r and rh
        dr_norm = np.linalg.norm(dr, ord=2) 
        drh_norm = np.linalg.norm(drh, ord=2)
        
        # Check convergence od r and rh, then update weights
        if dr_norm < eps and drh_norm < eps:
            error, errorh, dr, drh = model(inputs, training=True)
            break
        
        # If failure to convergence, break and print error
        if i >= nt_max-2: 
            print("Error at patch:", iter_)
            print(dr_norm, drh_norm)
            break
   
    
    error_list.append(model.calculate_total_error(error, errorh)) # Append errors

    # Decay learning rate         
    if iter_ % 40 == 39:
        model.k2 /= 1.015
    
    # Print moving average error
    if iter_ % 1000 == 999:  
        print("\n iter: "+str(iter_+1)+"/"+str(num_iter)+", Moving error:", np.mean(error_list[iter_-999:iter_]))
        
# Plot results
def moving_average(x, n=100) :
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

moving_average_error = moving_average(np.array(error_list))
plt.figure(figsize=(5, 3))
plt.ylabel("Error")
plt.xlabel("Iterations")
plt.plot(np.arange(len(moving_average_error)), moving_average_error)
plt.show()

# Plot Receptive fields of level 1
fig = plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(np.reshape(model.U[:, i], (16, 16)), cmap="gray")
    plt.axis("off")

plt.tight_layout()
fig.suptitle("Receptive fields of level 1", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF_level1.png")
plt.show()

# Plot Receptive fields of level 2
zeroPadding = np.zeros((80, 32))
U1 = np.concatenate((model.U, zeroPadding, zeroPadding))
U2 = np.concatenate((zeroPadding, model.U, zeroPadding))
U3 = np.concatenate((zeroPadding, zeroPadding, model.U))
U_ = np.concatenate((U1, U2, U3), axis = 1)
Uh_ = U_ @ model.Uh  

fig = plt.figure(figsize=(10, 5))
for i in range(24):
    plt.subplot(4, 6, i+1)
    plt.imshow(np.reshape(Uh_[:, i], (16, 26), order='F'), cmap="gray")
    plt.axis("off")

plt.tight_layout()
fig.suptitle("Receptive fields of level 2", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF_level2.png")
plt.show()
