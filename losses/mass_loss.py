import os
import torch
import numpy as np
from torchvision import transforms
from .ausm_losses import F_mass,F_mom
from .mass_maps import mass_conservation_calculate, momentum_conservation_calculate
from .f1_loss import F1_Loss

class CustomLoss(torch.nn.Module):
    
    def __init__(self, mean, std, loss_weights = (1. , 1. , 1., 1.), file=os.sys.stdout):
        super().__init__()       
        
        self.l_mse = loss_weights[0] / sum(loss_weights)
        self.l_mass= loss_weights[1] / sum(loss_weights) 
        self.l_gdl = loss_weights[2] / sum(loss_weights)
        self.l_mom = loss_weights[3] / sum(loss_weights)
        
       
        self.loss_mse  = 0.        
        self.loss_mass = 0. 
        self.loss_gdl  = 0.
        self.loss_mom  = 0.  
        self.file = file

        self.file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("mse",   "mass",   "gdl",   "mom", \
                                                                "w_mse", "w_mass", "w_gdl", "w_mom", \
                                                                "total"))
        
        self.inv_transform = transforms.Normalize([-m/s for (m,s) in zip(mean, std)], [1/s for s in std]) 

    
    def forward(self, output, target):

        b_size = output.shape[0]
        
        #output = output.nan_to_num()
        #target = target.nan_to_num()

        inv_output = self.inv_transform(output)#.nan_to_num()
        inv_target = self.inv_transform(target)#.nan_to_num() 
        
        #OUTPUT MASS & MOMENTUM 
        Rho = inv_output[:,0,:,:] # inverse transformation (output)
        U   = inv_output[:,1,:,:]
        V   = inv_output[:,2,:,:]
        P   = inv_output[:,3,:,:]
        T   = inv_output[:,4,:,:]
        Ma  = inv_output[:,5,:,:]
        
        mass_output =0 #F_mass(Rho, T, Ma)[:,1:-1,1:-1].nan_to_num()  
        #print(mass_output.shape, file=self.file)
        mom_x_output,mom_y_output = momentum_conservation_calculate(Rho, U, V, P)#[:,1:-1,1:-1]
        #print("mom x-y output shape : ",mom_x_output.shape,mom_y_output.shape, file=self.file)
        #TARGET MASS & MOMENTUM
        
        rho = inv_target[:,0,:,:] # inverse 
        u   = inv_target[:,1,:,:]
        v   = inv_target[:,2,:,:]
        p   = inv_target[:,3,:,:]
        
        #mass_target = mass_conservation_calculate(rho, u, v)[:,1:-1,1:-1]
        mom_x_target,mom_y_target = momentum_conservation_calculate(rho, u, v, p)#[:,1:-1,1:-1]
        #print("mom x-y output shape : ",mom_x_target.shape,mom_y_target.shape, file=self.file) 
        
        # GDL 
        Y_i = torch.abs(target - torch.roll(target, 1, 2))[:, :, 1:-1, 1:-1] # gradient of gt wrt i  
        Y_j = torch.abs(target - torch.roll(target, 1, 3))[:, :, 1:-1, 1:-1] # gradient of gt wrt j 
        
        Y_hat_i = torch.abs(output - torch.roll(output, 1, 2))[:, :, 1:-1, 1:-1] # gradient of prediction wrt i
        Y_hat_j = torch.abs(output - torch.roll(output, 1, 3))[:, :, 1:-1, 1:-1] # gradient of prediction wrt j 
        
        self.loss_gdl = torch.sum(torch.abs(Y_i - Y_hat_i) + torch.abs(Y_j - Y_hat_j)) / b_size
        
        self.loss_mse = torch.sum(torch.square(output - target)) / b_size  

        #self.loss_mass = torch.sum(torch.abs(mass_output - mass_target)) / b_size
        #self.loss_mass = torch.sum(torch.abs(mass_output)) / b_size # regularization wise 
        self.loss_mass =torch.tensor(0)
        #print(inv_target[0][0][100:125][100:125], file=self.file)
        self.loss_mom = torch.sum(torch.abs(mom_x_output-mom_x_target) + torch.abs(mom_y_output-mom_y_target) ) / b_size   
        
        loss = self.l_mse * self.loss_mse +  self.l_mass * self.loss_mass + self.l_gdl * self.loss_gdl + self.l_mom * self.loss_mom     
        
        self.file.write("%.3e\t%.3e\t%.3e\t%.3e\t" % (self.loss_mse.item(), 
                                                      self.loss_mass.item(), 
                                                      self.loss_gdl.item(), 
                                                      self.loss_mom.item()))
        self.file.write("%.3e\t%.3e\t%.3e\t%.3e\t" % ((self.l_mse * self.loss_mse).item(), 
                                                       (self.l_mass * self.loss_mass).item(), 
                                                       (self.l_gdl * self.loss_gdl).item(), 
                                                       (self.l_mom * self.loss_mom).item()))

        self.file.write("%.3e\n" % (loss.item(),))
        
        return loss
    
    
    def current_losses(self):
        return np.array([self.loss_mse.item() ,self.loss_mass.item() ,self.loss_gdl.item(), self.loss_mom.item()])
    
