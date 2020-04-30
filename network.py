# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

class OlshausenField1996Model:
    def __init__(self, dt=1, sigma2=1, sigma2_td=10):
        self.dt = dt
        self.inv_sigma2 = 1/sigma2 # 1 / sigma^2        
        
        self.eta = 1.0 # k_2: learning rate
        self.alpha = 0.14 # lambda / sigma
        
        self.num_units_level0 = 256
        self.num_units_level1 = 100
        self.batch_size = 100
        
        U = np.random.randn(self.num_units_level0, 
                            self.num_units_level1)
        self.U = U.astype(np.float32) * np.sqrt(2/(self.num_units_level0+self.num_units_level1))
                
        self.r = np.zeros((self.batch_size, self.num_units_level1))
    
    def initialize_states(self, inputs):
        self.r = np.array([self.U.T @ inputs[i] for i in range(self.batch_size)])
    
    def normalize_rows(self):
        self.U = 0.05*self.U / np.linalg.norm(self.U, ord=2, axis=1, keepdims=True)

    # fun(x, *args) -> float
    def _cost(self, r_, U_, input_):
        error = input_ - U_ @ r_ 
        
        sigma = np.std(input_)
        rs = r_ / sigma
        S_r = self.alpha * sigma * np.sum(np.log(1 + rs**2))
        return np.sum(error**2) + S_r

    def _gradient(self, r_, U_, input_):
        error = input_ - U_ @ r_ 

        sigma = np.std(input_)
        rs = r_ / sigma
        g_r = self.alpha * rs / (1 + rs**2) 

        return - U_.T @ error + g_r
    
    def _optimize_coefficients(self, U_, input_):
        r_init = U_.T @ input_

        res = opt.minimize(fun=self._cost, 
                           x0=r_init,
                           args=(U_, input_),
                           method='CG',
                           jac=self._gradient,
                           options={'gtol': 1e-2, 'disp': False})
        return res

    def calculate_total_error(self, error):
        recon_error = np.mean(error**2)
        sparsity_r = self.alpha*np.mean(self.r**2) 
        return recon_error + sparsity_r
        
    def __call__(self, inputs, training=False):
        # Updates                
        for i in range(self.batch_size):
            self.r[i] = self._optimize_coefficients(self.U, inputs[i]).get('x')

        fx = np.array([self.U @ self.r[i] for i in range(self.batch_size)]) # (3, 256)
        error = inputs - fx # (3, 256)
        
        if training:  
            dU = np.mean(np.array([np.outer(error[i], self.r[i]) for i in range(self.batch_size)]), axis=0)            
            self.U += self.eta * dU
            
        return error
            
            
            