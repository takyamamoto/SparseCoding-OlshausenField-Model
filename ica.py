# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from tqdm import tqdm 
import scipy.io as sio

# datasets from http://www.rctn.org/bruno/sparsenet/
# mat_images = sio.loadmat('datasets/IMAGES.mat')
# imgs = mat_images['IMAGES']
mat_images_raw = sio.loadmat('datasets/IMAGES_RAW.mat')
imgs_raw = mat_images_raw['IMAGESr']

# Simulation constants
H, W, num_images = imgs_raw.shape

num_patches = 15000
patchs_list = []
w, h = 16, 16 # patch size

# generate patches
for i in tqdm(range(num_patches)):
    i = np.random.randint(0, num_images)
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-w-1)
    beginy = np.random.randint(0, H-h-1)
    img_clopped = imgs_raw[beginy:beginy+h, beginx:beginx+w, i]
    patchs_list.append(img_clopped.flatten())

patches = np.array(patchs_list)

# perform ICA
print("perform ICA")
n_comp = 100
ica = FastICA(n_components=n_comp)
ica.fit(patches)
ica_filters = ica.components_

# plot filters
plt.figure(figsize=(6,6))
plt.subplots_adjust(hspace=0.1, wspace=0.1)
for i in tqdm(range(n_comp)):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(ica_filters[i], (w, h)), cmap="gray")
    plt.axis("off")
plt.suptitle("ICA", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("ICA.png")
plt.show()

# perform PCA
print("perform PCA")
pca = PCA(n_components=n_comp)
pca.fit(patches)
pca_filters = pca.components_

# plot filters
plt.figure(figsize=(6,6))
plt.subplots_adjust(hspace=0.1, wspace=0.1)
for i in tqdm(range(n_comp)):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(pca_filters[i], (w, h)), cmap="gray")
    plt.axis("off")
plt.suptitle("PCA", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("PCA.png")
plt.show()

