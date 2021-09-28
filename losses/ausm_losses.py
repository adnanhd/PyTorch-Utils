import torch

def a(T):
	gamma = 1.4 
	R = 287.058
	return torch.sqrt(gamma*R*T)


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


def M_R(M):
	M_plus_i = M_plus(M)
	M_minus_i_plus_1 = M_minus(shift_L(M))

	return M_plus_i + M_minus_i_plus_1


def M_L(M):
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


def F_mass(Rho,T,M):

	return M_R(M) * Phi_R(Rho*a(T)) - M_L(M) * Phi_L(Rho*a(T))


def F_mom(Rho,U,V,P,T,M):
	eq1 = M_R(M) * Phi_R(Rho*a(T)*U) + P_R(P,M)
	eq2 = M_L(M) * Phi_L(Rho*a(T)*U) + P_L(P,M)

	eq3 = M_R(M) * Phi_R(Rho*a(T)*V)  
	eq4 = M_L(M) * Phi_L(Rho*a(T)*V)

	return eq1 - eq2 + eq3 - eq4














