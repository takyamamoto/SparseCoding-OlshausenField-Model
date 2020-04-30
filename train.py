# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm
import scipy.io as sio

np.random.seed(0)

# Preprocess of inputs
num_iter = 4000

# datasets from http://www.rctn.org/bruno/sparsenet/
mat_images = sio.loadmat('datasets/IMAGES.mat')
imgs = mat_images['IMAGES']
#mat_images_raw = sio.loadmat('datasets/IMAGES_RAW.mat')
#imgs_raw = mat_images_raw['IMAGESr']
    
# Define model
model = network.OlshausenField1996Model()

# Simulation constants
H, W, num_images = imgs.shape
input_scale = 5 # scale factor of inputs
error_list = [] # List to save errors
sz = 16 # size of batch edge
batch_size = 100

for iter_ in tqdm(range(num_iter)):
    # Get images randomly
    idx = np.random.randint(0, num_images)
    img = imgs[:, :, idx]
    
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-sz-1, batch_size)
    beginy = np.random.randint(0, H-sz-1, batch_size)
    
    inputs = np.array([img[beginy[i]:beginy[i]+sz,
                           beginx[i]:beginx[i]+sz].flatten() for i in range(batch_size)])
    
    # Clop three inputs
    inputs = (inputs - np.mean(inputs, axis=0)) #* input_scale
    
    # Reset states
    model.initialize_states(inputs)
    #model.normalize_rows()
    
    # Input an image patch until latent variables are converged 
    error = model(inputs, training=True)
          
    error_list.append(model.calculate_total_error(error)) # Append errors

    # Decay learning rate         
    if iter_ % 40 == 39:
        model.eta /= 1.015
    
    # Print moving average error
    if iter_ % 100 == 99:  
        print("\n iter: "+str(iter_+1)+"/"+str(num_iter)+", Moving error:", np.mean(error_list[iter_-999:iter_]))

plt.figure(figsize=(5, 3))
plt.ylabel("Error")
plt.xlabel("Iterations")
plt.plot(np.arange(len(error_list)), np.array(error_list))
plt.savefig("error.png")
plt.show()

# Plot Receptive fields
fig = plt.figure(figsize=(8, 8))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(model.U[:, i], (16, 16)), cmap="gray")
    plt.axis("off")

plt.tight_layout()
fig.suptitle("Receptive fields", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF.png")
plt.show()