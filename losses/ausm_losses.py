import torch
import pdb
import wandb


def a(T):
	gamma = 1.4 
	R = 287.058
	return (gamma*R*T).abs().sqrt()


def shift_L(t):
	# i=i+1
	D=len(t.shape)

	return torch.roll(t,-1,D-1)


def shift_R(t):
	# i=i-1
	D=len(t.shape)

	return torch.roll(t,1,D-1)
	

def M_plus(M):
	m_lt1 = (torch.abs(M) <=1)
	m_gt1 = (torch.abs(M) > 1)

	M_lt = (0.25)*torch.square(M+1) * m_lt1
	M_gt = M * (M>0) * m_gt1

	return M_lt + M_gt


def M_minus(M):
	m_lt1 = (torch.abs(M) <=1)
	m_gt1 = (torch.abs(M) > 1)

	M_lt = -(0.25)*torch.square(M-1) * m_lt1
	M_gt = M * (M<0) * m_gt1

	return M_lt + M_gt	


def M_R(M):  # Right side
	M_plus_i = M_plus(M)
	M_minus_i_plus_1 = M_minus(shift_L(M))

	return M_plus_i + M_minus_i_plus_1


def M_L(M): # left side
	M_plus_i_minus_1 = M_plus(shift_R(M))
	M_minus_i = M_minus(M)

	return M_plus_i_minus_1 + M_minus_i


def Phi_R(t):
	m_R = M_R(t)
	gt0 = m_R>0
	lt0 = m_R<0

	phi_gt = t*gt0
	phi_lt = shift_L(t)*lt0 

	return (phi_gt + phi_lt)
	 

def Phi_L(t):
	m_L = M_L(t)
	
	gt0 = m_L>0
	lt0 = m_L<0

	phi_gt = shift_R(t)*lt0 
	phi_lt = t*gt0

	return (phi_gt + phi_lt)


def P_plus(P,M):
	m_lt1 = (torch.abs(M) <=1)

	pplus_lt1 = 0.5 * P * (1+M) * m_lt1
	pplus_gt1 = P * (M>1)

	return pplus_lt1 + pplus_gt1


def P_minus(P,M):
	m_lt1 = (torch.abs(M) <=1)
	
	pminus_lt1 = 0.5 * P * (1-M) * m_lt1
	pminus_gt1 = - P * (M<-1)

	return pminus_lt1 + pminus_gt1	


def P_R(P,M):
	pplus_i = P_plus(P,M)
	pminus_i_plus_1 = P_minus(shift_L(P),shift_L(M))

	return pplus_i + pminus_i_plus_1


def P_L(P,M):
	pplus_i_minus_1 = P_plus(shift_R(P),shift_R(M))
	pminus = P_minus(P,M)

	return pplus_i_minus_1 + pminus

class MassConservationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, ground=None):
        """
        Rho = output[:,0,:,:] 
        U   = output[:,1,:,:]
        V   = output[:,2,:,:]
        P   = output[:,3,:,:]
        T   = output[:,4,:,:]
        M   = output[:,5,:,:]
        """
        result = self._F_mass(*map(lambda i: output[:,i,:,:], (0, 4, 5)))
        print(result.mean(), result.sum())

        if ground is not None:
            result = result - self._F_mass(*map(lambda i: output[:,i,:,:], (0, 4, 5)))

        return result.sum()
        
    @staticmethod
    def _F_mass(Rho, T, M):
        return M_R(M) * Phi_R(Rho*a(T)) - M_L(M) * Phi_L(Rho*a(T))


class MomentumConservationLoss(torch.nn.Module):
    wandb = False
    debug = False
    def __init__(self):
        super().__init__()
    
    def forward(self, output, ground=None):
        """
        Rho = output[:,0,:,:] 
        U   = output[:,1,:,:]
        V   = output[:,2,:,:]
        P   = output[:,3,:,:]
        T   = output[:,4,:,:]
        M   = output[:,5,:,:]
        """

        if self.debug and output.isnan().any():
            pdb.set_trace()
        
        result = self._F_mom(*map(lambda i: output[:,i,:,:], range(6)))
        
        if self.debug and result.isnan().any():
            print("output fucked up")
            pdb.set_trace()
        
        if ground is not None:
            result = result - self._F_mom(*map(lambda i: ground[:,i,:,:], range(6)))
            
            if self.debug and ground.isnan().any():
                pdb.set_trace()
            
            if self.debug and result.isnan().any():
                print("ground fucked up")
                pdb.set_trace()

        return result.sum()

    @classmethod
    def _F_mom(cls, Rho, U, V, P, T, M):
        eq1 = M_R(M) * Phi_R(Rho * a(T) * U) + P_R(P, M)
        eq2 = M_L(M) * Phi_L(Rho * a(T) * U) + P_L(P, M)
        
        eq3 = M_R(M) * Phi_R(Rho * a(T) * V)  
        eq4 = M_L(M) * Phi_L(Rho * a(T) * V)

        flag = False

        if cls.debug and eq1.isnan().any():
            print("eq1 is fucked up")
            flag = True
        elif cls.wandb:
            wandb.log({"eq1_loss": eq1.mean()})

        if cls.debug and eq2.isnan().any():
            print("eq2 is fucked up")
            flag = True
        elif cls.wandb:
            wandb.log({"eq2_loss": eq2.mean()})

        if cls.debug and eq3.isnan().any():
            print("eq3 is fucked up")
            flag = True
        elif cls.wandb:
            wandb.log({"eq3_loss": eq3.mean()})

        if cls.debug and eq4.isnan().any():
            print("eq4 is fucked up")
            flag = True
        elif cls.wandb:
            wandb.log({"eq4_loss": eq4.mean()})

        if cls.debug and flag:
            pdb.set_trace()
        
        return (eq1 - eq2 + eq3 - eq4).sum(dim=-3)
