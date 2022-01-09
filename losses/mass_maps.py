import torch


# roll  1: image yukari, saga kayiyor 
# roll -1: image asagi, sola kayiyor

def e(t):
	D= len(t.shape)
	return (t+ torch.roll(t,-1,D-1))/2		

def w(t):
	D=len(t.shape)
	return (t+ torch.roll(t, 1,D-1) )/2 

def n(t): 
	D = len(t.shape)
	return (t+ torch.roll(t, -1,D-2) ) /2 

def s(t): 
	D = len(t.shape)
	return (t+ torch.roll(t, 1,D-2) ) /2 

def four_walls(t):
	return e(t), w(t) , n(t) , s(t)


def mass_conservation_calculate(rho, u, v): # all tensors of dimension (....H,W)

    A_L = A_R = 1/256
    A_N = A_S = 2/256
    
    D = len(rho.shape)
    
    h = D-2 # bi onceki parametre, yukari asagi
    w = D-1 # son parametre, saga sola 


    rho_r,rho_l,rho_n,rho_s = four_walls(rho)

    u_l = (u + torch.roll(u, 1,w)) / 2
    u_r = (u + torch.roll(u,-1,w)) / 2
    
    v_n = (v + torch.roll(v,-1,h)) / 2
    v_s = (v + torch.roll(v, 1,h)) / 2
    
    m_l = rho_l * u_l * A_L
    m_r = rho_r * u_r * A_R
    m_n = rho_n * v_n * A_N
    m_s = rho_s * v_s * A_S

    mass_conserved = (-m_l + m_r) + (m_n - m_s)
    
    return mass_conserved



def momentum_conservation_calculate(rho, u, v, p):
    
    rho_e,rho_w,rho_n,rho_s = four_walls(rho)
    u_e , u_w , u_n , u_s   = four_walls(u)
    v_e , v_w , v_n , v_s   = four_walls(v)
    p_e , p_w , p_n , p_s   = four_walls(p)
    
    
    ## X MOMENTUM =  [ rho_e * (u_e)^2 + P_e ] - [ rho_w * (u_w)^2 + P_w ]  + rho_n * u_n * v_n - rho_s * u_s * v_s 
    
    f1 = rho_e * u_e * u_e + p_e #formuldeki ilk parantez
    f2 = rho_w * u_w * u_w + p_w #2.parantez
    f3 = rho_n * u_n * v_n  
    f4 = rho_s * u_s * v_s
    
    x_momentum = (f1 - f2) + (f3 - f4) 
    
    ## Y MOMENTUM =  [ rho_n * (u_n)^2 + P_n ] - [ rho_s * (u_s)^2 + P_s ]  + rho_e * u_e * v_e - rho_w * u_w * v_w
    
    f1 = rho_n * v_n*v_n + p_n 
    f2 = rho_s * v_s*v_s + p_s 
    f3 = rho_e * u_e * v_e # can be rho_e * u_e * v_e ?
    f4 = rho_w * u_w * v_w # can be rho_w * u_w * v_w ?
    
    y_momentum = (f1 - f2) + (f3 - f4) 
    
    return x_momentum + y_momentum
