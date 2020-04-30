# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm
import scipy.io as sio

np.random.seed(0)

# datasets from http://www.rctn.org/bruno/sparsenet/
mat_images = sio.loadmat('datasets/IMAGES.mat')
imgs = mat_images['IMAGES']
    
# Simulation constants
H, W, num_images = imgs.shape
num_iter = 500 # number of iterations
nt_max = 1000 # Maximum number of simulation time
batch_size = 250 # Batch size

sz = 16 # image patch size
num_units = 100 # number of neurons (units)

eps = 1e-2 # small value which determines convergence
error_list = [] # List to save errors

# Define model
model = network.OlshausenField1996Model(num_inputs=sz**2, num_units=num_units,
                                        batch_size=batch_size)

# Run simulation
for iter_ in tqdm(range(num_iter)):
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-sz, batch_size)
    beginy = np.random.randint(0, H-sz, batch_size)

    inputs_list = []

    # Get images randomly
    for i in range(batch_size):        
        idx = np.random.randint(0, num_images)
        img = imgs[:, :, idx]
        clop = img[beginy[i]:beginy[i]+sz, beginx[i]:beginx[i]+sz].flatten()
        inputs_list.append(clop - np.mean(clop))
        
    inputs = np.array(inputs_list) # Input image patches
    
    model.initialize_states() # Reset states
    model.normalize_rows() # Normalize weights
    
    # Input an image patch until latent variables are converged 
    r_tm1 = model.r # set previous r (t minus 1)

    for t in range(nt_max):
        # Update r without update weights 
        error, r = model(inputs, training=False)
        dr = r - r_tm1 

        # Compute norm of r
        dr_norm = np.linalg.norm(dr, ord=2) / (eps + np.linalg.norm(r_tm1, ord=2))
        r_tm1 = r # update r_tm1
        
        # Check convergence of r, then update weights
        if dr_norm < eps:
            error, r = model(inputs, training=True)
            break
        
        # If failure to convergence, break and print error
        if t >= nt_max-2: 
            print("Error at patch:", iter_)
            print(dr_norm)
            break
   
    error_list.append(model.calculate_total_error(error)) # Append errors

    # Print moving average error
    if iter_ % 100 == 99:  
        print("\n iter: "+str(iter_+1)+"/"+str(num_iter)+", Moving error:",
              np.mean(error_list[iter_-99:iter_]))

# Plot error
plt.figure(figsize=(5, 3))
plt.ylabel("Error")
plt.xlabel("Iterations")
plt.plot(np.arange(len(error_list)), np.array(error_list))
plt.tight_layout()
plt.savefig("error.png")
plt.show()

# Plot Receptive fields
fig = plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.1, wspace=0.1)
for i in tqdm(range(num_units)):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(model.Phi[:, i], (sz, sz)), cmap="gray")
    plt.axis("off")

fig.suptitle("Receptive fields", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF.png")
plt.show()