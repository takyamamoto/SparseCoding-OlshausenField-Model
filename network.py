# -*- coding: utf-8 -*-

import numpy as np


class RaoBallard1999Model:
    def __init__(self, dt=1, sigma2=1, sigma2_td=10):
        self.dt = dt
        self.inv_sigma2 = 1/sigma2 # 1 / sigma^2        
        self.inv_sigma2_td = 1/sigma2_td # 1 / sigma_td^2
        
        self.k1 = 0.3 # k_1: update rate
        self.k2 = 0.2 # k_2: learning rate
        
        self.lam = 0.02 # sparsity rate
        self.alpha = 1
        self.alphah = 0.05
        
        self.num_units_level0 = 256
        self.num_units_level1 = 32
        self.num_units_level2 = 128
        self.num_level1 = 3
        
        U = np.random.randn(self.num_units_level0, 
                            self.num_units_level1)
        Uh = np.random.randn(int(self.num_level1*self.num_units_level1),
                             self.num_units_level2)
        self.U = U.astype(np.float32) * np.sqrt(2/(self.num_units_level0+self.num_units_level1))
        self.Uh = Uh.astype(np.float32) * np.sqrt(2/(int(self.num_level1*self.num_units_level1)+self.num_units_level2)) 
                
        self.r = np.zeros((self.num_level1, self.num_units_level1))
        self.rh = np.zeros((self.num_units_level2))
    
    def initialize_states(self, inputs):
        self.r = np.array([self.U.T @ inputs[i] for i in range(self.num_level1)])
        self.rh = self.Uh.T @ np.reshape(self.r, (int(self.num_level1*self.num_units_level1)))
    
    def calculate_total_error(self, error, errorh):
        recon_error = self.inv_sigma2*np.sum(error**2) + self.inv_sigma2_td*np.sum(errorh**2)
        sparsity_r = self.alpha*np.sum(self.r**2) + self.alphah*np.sum(self.rh**2)
        sparsity_U = self.lam*(np.sum(self.U**2) + np.sum(self.Uh**2))
        return recon_error + sparsity_r + sparsity_U
        
    def __call__(self, inputs, training=False):
        # inputs : (3, 256)
        r_reshaped = np.reshape(self.r, (int(self.num_level1*self.num_units_level1))) # (96)

        fx = np.array([self.U @ self.r[i] for i in range(self.num_level1)]) # (3, 256)
        #fx = np.array([np.tanh(self.U @ self.r[i]) for i in range(self.num_level1)]) # (3, 256)

        fxh = self.Uh @ self.rh # (96, )
        #fxh = np.tanh(self.Uh @ self.rh) # (96, )
        
        #dfx = 1 - fx**2 # (3, 256)
        #dfxh = 1 - fxh**2 # (96,)
        
        error = inputs - fx # (3, 256)
        errorh = r_reshaped - fxh # (96, ) 
        errorh_reshaped = np.reshape(errorh, (self.num_level1, self.num_units_level1)) # (3, 32)

        #dfx_error = dfx * error # (3, 256)
        #dfxh_errorh = dfxh * errorh # (96, )
        
        g_r = self.alpha * self.r / (1 + self.r**2) # (3, 32)
        g_rh = self.alphah * self.rh / (1 + self.rh**2) # (64, )
        #g_r = self.alpha * self.r  # (3, 32)
        #g_rh = self.alphah * self.rh # (64, )

        dr = self.inv_sigma2 * np.array([self.U.T @ error[i] for i in range(self.num_level1)])\
            - self.inv_sigma2_td * errorh_reshaped - g_r
        drh = self.inv_sigma2_td * self.Uh.T @ errorh - g_rh
        
        """
        dr = self.inv_sigma2 * np.array([self.U.T @ dfx_error[i] for i in range(self.num_level1)])\
            - self.inv_sigma2_td * errorh_reshaped - g_r
        drh = self.inv_sigma2_td * self.Uh.T @ dfxh_errorh - g_rh
        """
        
        dr = self.k1 * dr
        drh = self.k1 * drh
        
        # Updates                
        self.r += dr
        self.rh += drh
        
        if training:  
            """
            dU = self.inv_sigma2 * np.sum(np.array([np.outer(dfx_error[i], self.r[i]) for i in range(self.num_level1)]),axis=0)\
                - 3*self.lam * self.U
            dUh = self.inv_sigma2_td * np.outer(dfxh_errorh, self.rh)\
                - self.lam * self.Uh
            """
            dU = self.inv_sigma2 * np.sum(np.array([np.outer(error[i], self.r[i]) for i in range(self.num_level1)]), axis=0)\
                - 3*self.lam * self.U
            dUh = self.inv_sigma2_td * np.outer(errorh, self.rh)\
                - self.lam * self.Uh
            
            self.U += self.k2 * dU
            self.Uh += self.k2 * dUh
            
        return error, errorh, dr, drh
            
            
            