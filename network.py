# -*- coding: utf-8 -*-

import numpy as np

class OlshausenField1996Model:
    def __init__(self, num_inputs, num_units, batch_size,
                 lr_r=1e-2, lr_Phi=1e-2, lmda=5e-3):
        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.lmda = lmda # regularization parameter
        
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size
        
        # Weights
        Phi = np.random.randn(self.num_inputs, self.num_units).astype(np.float32)
        self.Phi = Phi * np.sqrt(1/self.num_units)

        # activity of neurons
        self.r = np.zeros((self.batch_size, self.num_units))
    
    def initialize_states(self):
        self.r = np.zeros((self.batch_size, self.num_units))
        
    def normalize_rows(self):
        self.Phi = self.Phi / np.maximum(np.linalg.norm(self.Phi, ord=2, axis=0, keepdims=True), 1e-8)

    # thresholding function of S(x)=|x|
    def soft_thresholding_func(self, x, lmda):
        return np.maximum(x - lmda, 0) - np.maximum(-x - lmda, 0)

    # thresholding function of S(x)=ln(1+x^2)
    def ln_thresholding_func(self, x, lmda):
        f = 9*lmda*x - 2*np.power(x, 3) - 18*x
        g = 3*lmda - np.square(x) + 3
        h = np.cbrt(np.sqrt(np.square(f) + 4*np.power(g, 3)) + f)
        two_croot = np.cbrt(2) # cubic root of two
        return (1/3)*(x - h / two_croot + two_croot*g / (1e-8+h))

    # thresholding function https://arxiv.org/abs/2003.12507
    def cauchy_thresholding_func(self, x, lmda):
        f = 0.5*(x + np.sqrt(np.maximum(x**2 - lmda,0)))
        g = 0.5*(x - np.sqrt(np.maximum(x**2 - lmda,0)))
        return f*(x>=lmda) + g*(x<=-lmda) 

    def calculate_total_error(self, error):
        recon_error = np.mean(error**2)
        sparsity_r = self.lmda*np.mean(np.abs(self.r)) 
        return recon_error + sparsity_r
        
    def __call__(self, inputs, training=True):
        # Updates                
        error = inputs - self.r @ self.Phi.T
        
        r = self.r + self.lr_r * error @ self.Phi
        self.r = self.soft_thresholding_func(r, self.lmda)
        #self.r = self.cauchy_thresholding_func(r, self.lmda)
        
        if training:  
            error = inputs - self.r @ self.Phi.T
            dPhi = error.T @ self.r
            self.Phi += self.lr_Phi * dPhi
            
        return error, self.r
