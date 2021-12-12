import os
import torch
import numpy as np
from torchvision import transforms
from .mass_maps import mass_conservation_calculate, momentum_conservation_calculate
from .f1_loss import F1_Loss
from .ausm_losses import MomentumConservationLoss, MassConservationLoss
from torch.nn import MSELoss

class CustomLoss(torch.nn.Module):
    _MSE = 0
    _MAS = 1
    _MOM = 2
    def __init__(self, mse_weight=1. , mass_weight=0., mom_weight=0., dtype=None, device=None,
                       mean=None, std=None, file=os.sys.stdout, verbose=False):
        super().__init__()       
        
        self.file = file
        self.verbose = verbose
        weights = float(mass_weight + mom_weight + mse_weight)
        self.weights = torch.empty(3, device=device, dtype=dtype)

        self.weights[self._MSE] = torch.tensor(mse_weight  / weights if mse_weight  else False)
        self.weights[self._MAS] = torch.tensor(mass_weight / weights if mass_weight else False)
        self.weights[self._MOM] = torch.tensor(mom_weight  / weights if mom_weight  else False)
        
       
        self.mse_loss = MSELoss()
        self.mas_loss = MassConservationLoss()
        self.mom_loss = MomentumConservationLoss()

        if self.verbose:
            self.file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("total",
                                                 "mse", "mom", "mass", 
                                                 "w_mse", "w_mom", "w_mass"))
        
        if mean and std and False:
            self.inv_transform = transforms.Normalize(
                    [-m/s for (m,s) in zip(mean, std)], [1/s for s in std]) 
        else:
            self.inv_transform = None

    
    def forward(self, output, ground):
        loss = torch.empty(3, dtype=output.dtype, device=output.device)
        
        if self.inv_transform:
            output = self.inv_transform(output)
            ground = self.inv_transform(ground)


        loss[self._MSE] = self.mse_loss(output, ground)
        loss[self._MOM] = self.mom_loss(output, ground=None)
        loss[self._MAS] = self.mas_loss(output, ground=None).detach()

        w_loss = loss * self.weights

        result = w_loss[self._MSE] + w_loss[self._MOM]

        if self.verbose:
            self.file.write(
                    "%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n" % (result.item(), *loss, *w_loss))
        
        return result
    
