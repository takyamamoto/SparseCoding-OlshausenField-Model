# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:08:04 2020

@author: Yamamoto T
"""

#https://peekaboo-vision.blogspot.com/2012/04/learning-gabor-filters-with-ica-and.html

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.decomposition import FastICA, PCA
from tqdm import tqdm 

imgdirpath = "./images_preprocessed/"
imglist = []

for i in range(10):
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    #img_loaded = cv2.imread(filepath)[:, :, 0]
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float64)
    img_loaded = (img_loaded - np.mean(img_loaded)) / np.std(img_loaded)
    imglist.append(img_loaded)

imgs = np.array(imglist)
"""
plt.figure(figsize=(5,10))
for i in tqdm(range(10)):
    plt.subplot(2, 5, i+1)
    plt.imshow(imgs[:,:,i], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()
"""
num_images, W, H = imgs.shape

n_patchs = 20000
patchs_list = []
w, h = 16, 16

for i in tqdm(range(n_patchs)):
    i = np.random.randint(0, num_images)
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-w-1)
    beginy = np.random.randint(0, H-h-1)
    img_clopped = imgs[i, beginy:beginy+h, beginx:beginx+w]
    patchs_list.append(img_clopped.flatten())

patchs = np.array(patchs_list)
# perform ICA
n_comp = 64
ica = FastICA(n_components=n_comp)
ica.fit(patchs)
ica_filters = ica.components_

# plot filters
plt.figure(figsize=(6,6))
for i in tqdm(range(n_comp)):
    plt.subplot(8, 8, i+1)
    plt.imshow(np.reshape(ica_filters[i], (w, h)), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("ICA", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.show()

pca = PCA(n_components=n_comp)
pca.fit(patchs)
pca_filters = pca.components_
# plot filters
plt.figure(figsize=(6,6))
for i in tqdm(range(n_comp)):
    plt.subplot(8, 8, i+1)
    plt.imshow(np.reshape(pca_filters[i], (w, h)), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("PCA", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.show()

