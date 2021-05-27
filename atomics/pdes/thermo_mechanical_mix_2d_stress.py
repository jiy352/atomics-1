import dolfin as df
import pygmsh

def get_residual_form(u, v, rho_e, T, T_hat, KAPPA, k, alpha, mode='plane_stress', method='RAMP'):
    if method=='RAMP':
        C = rho_e/(1 + 8. * (1. - rho_e))
    else:
        C = rho_e**3

    E = k * C 
    # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio


    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    if mode == 'plane_stress':
        lambda_ = 2*mu*lambda_/(lambda_+2*mu)

    # Th = df.Constant(7)
    I = df.Identity(len(u))
    # w_ij = 0.5 * (df.grad(u) + df.grad(u).T) - alpha * I * T
    # change May 23 penalize on alpha
    T_0 = df.Constant(20)
    w_ij = 0.5 * (df.grad(u) + df.grad(u).T) - C*alpha * I * (T-T_0)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    d = len(u)
    # kappa = alpha*(2*mu + 3*lambda_)
    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij 
    # eps_u = df.sym(df.grad(u))
    # sigm = (lambda_*df.tr(eps_u)-kappa*T)*df.Identity(d) + 2*mu*eps_u 

    a = df.inner(sigm, v_ij) * df.dx + \
        df.dot(C*KAPPA* df.grad(T-T_0),  df.grad(T_hat)) * df.dx
    print("get a-------")
    
    return a


