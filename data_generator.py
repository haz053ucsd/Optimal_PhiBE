import torch
from torch.distributions import MultivariateNormal
import scipy.linalg
import numpy as np
from math import sqrt
torch.set_default_dtype(torch.float64)

# 1D LQR
def linear_dyn_generator_stochastic_const_act_exact(A, B, sig, running_b, I, m, dt, bd_low_s, bd_upper_s):
    # return shape (m, I), generate data for computing V in the case where diffusion coefficient is a constant (first order)

    init_value = (bd_upper_s - bd_low_s) * torch.rand(m, 1) + bd_low_s
    res = torch.zeros((m, I))
    res[:, 0] = init_value.squeeze()

    prev = res[:, 0]
    a_temp = running_b * prev
    act_mat = torch.zeros((m, I))
    act_mat[:, 0] = a_temp
    
    A = torch.tensor([A], dtype=torch.float64)
    B = torch.tensor([B], dtype=torch.float64)
    for i in range(1, I):

        mean = torch.exp(A * dt) * prev + B * a_temp / A * (torch.exp(A * dt) - 1)

        var = sig**2 / (2 * A) * (torch.exp(2 * A * dt) - 1)

        prev = mean + torch.randn(m) * torch.sqrt(var)

        a_temp = running_b * prev
        act_mat[:, i] = a_temp

        res[:, i] = prev

    return res, act_mat

def linear_dyn_generator_stochastic_const_act_exact_2nd(A, B, sig, running_b, I, m, dt, bd_low_s, bd_upper_s):
    # return shape (m, I), generate data for computing V in the case where diffusion coefficient is a constant (2nd order)

    init_value = (bd_upper_s - bd_low_s) * torch.rand(m, 1) + bd_low_s
    res = torch.zeros((m, I))
    res[:, 0] = init_value.squeeze()

    prev = res[:, 0]
    a_temp = running_b * prev
    act_mat = torch.zeros((m, I))
    act_mat[:, 0] = a_temp
    A = torch.tensor([A], dtype=torch.float64)
    B = torch.tensor([B], dtype=torch.float64)
    for i in range(1, I):

        mean = torch.exp(A * dt) * prev + B * a_temp / A * (torch.exp(A * dt) - 1)

        var = sig**2 / (2 * A) * (torch.exp(2 * A * dt) - 1)

        prev = mean + torch.randn(m) * torch.sqrt(var)


        res[:, i] = prev

        if i % 2 == 0:
            a_temp = running_b * prev
        act_mat[:, i] = a_temp

    return res, act_mat

def Q_dyn_generator_const_act_exact(A, B, sig, I, m_Q, dt, bd_low_s, bd_upper_s,
                                    bd_low_b, bd_upper_b):
    # return shape (m_Q, I), (m_Q, I),
    # generate data for computing Q in the case where diffusion coefficient is a constant for Phibe (1st order)
    b_temp_tensor = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m_Q)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m_Q, 1) + bd_low_s

    res = torch.zeros((m_Q, I))
    res[:, 0] = init_value.squeeze()
    act_mat = torch.zeros((m_Q, I))
    act_mat[:, 0] = b_temp_tensor * res[:, 0]

    prev = res[:, 0]
    act_temp = act_mat[:, 0]
    A = torch.tensor([A], dtype=torch.float64)
    B = torch.tensor([B], dtype=torch.float64)
    for i in range(1, I):
        mean = torch.exp(A * dt) * prev + B * act_temp / A * (torch.exp(A * dt) - 1)
        var = sig**2 / (2 * A) * (torch.exp(2 * A * dt) - 1)

        prev = mean + torch.randn(m_Q) * torch.sqrt(var)

        res[:, i] = prev

        act_temp = b_temp_tensor * prev

        act_mat[:, i] = act_temp

    return res, act_mat

def Q_dyn_generator_const_act_2nd_exact(A, B, sig, I, m_Q, dt, bd_low_s, bd_upper_s,
                                        bd_low_b, bd_upper_b):
    # return shape (m_Q, I), (m_Q, I)
    # generate data for computing second order Q in the case where diffusion coefficient is a constant for Phibe (2nd order)
    b_temp_tensor = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m_Q)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m_Q, 1) + bd_low_s

    res = torch.zeros((m_Q, I))
    res[:, 0] = init_value.squeeze()
    act_mat = torch.zeros((m_Q, I))
    act_mat[:, 0] = b_temp_tensor * res[:, 0]
    prev = res[:, 0]
    act_temp = act_mat[:, 0]
    A = torch.tensor([A], dtype=torch.float64)
    B = torch.tensor([B], dtype=torch.float64)
    for i in range(1, I):
        mean = torch.exp(A * dt) * prev + B * act_temp / A * (torch.exp(A * dt) - 1)
        var = sig**2 / (2 * A) * (torch.exp(2 * A * dt) - 1)

        prev = mean + torch.randn(m_Q) * torch.sqrt(var)

        res[:, i] = prev

        if i % 2 == 0:
            act_temp = b_temp_tensor * prev
        act_mat[:, i] = act_temp
    return res, act_mat

def linear_dyn_generator_stochastic_Q_exact(A, B, sig, running_b,
                                            I, m, dt, bd_low_s, bd_upper_s, bd_low_a, bd_upper_a):
    # return shape (m, I)
    # generate data for computing Q in the case where diffusion coefficient is a constant for RL
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m, 1) + bd_low_s

    res = torch.zeros((m, I))
    res[:, 0] = init_value.squeeze()

    act = torch.zeros((m, I))

    prev = res[:, 0]
    a_temp = (bd_upper_a - bd_low_a) * torch.rand(m) + bd_low_a

    act[:, 0] = a_temp[:]

    A = torch.tensor([A], dtype=torch.float64)
    B = torch.tensor([B], dtype=torch.float64)
    for i in range(1, I):
        mean = torch.exp(A * dt) * prev + B * a_temp / A * (torch.exp(A * dt) - 1)
        var = sig**2 / (2 * A) * (torch.exp(2 * A * dt) - 1)

        prev = mean + torch.randn(m) * torch.sqrt(var)

        res[:, i] = prev

        a_temp = running_b * prev

        act[:, i] = a_temp
    return res, act

# 2D LQR
def one_step_2D_exact(prev, A_1, B_1, C_A, sig, action, dt, dim):
    #  compute one step evolution 
    #  prev (m, dim), A (dim, dim), B (dim, dim), action (m, dim)
    #  return (m, dim)
    if sig != 0:
        mean_coe = A_1 * dt + torch.eye(dim)
        mean = torch.einsum('ij,kj->ki', mean_coe, prev) + dt * torch.einsum('ij,kj->ki', B_1, action)  # (m ,dim)
        covar = (sig**2 * C_A * dt).unsqueeze(0).repeat(prev.shape[0], 1, 1)  # (m, dim, dim)
        # print(C_A)
        mvn = MultivariateNormal(mean, covariance_matrix=covar)
        samples = mvn.sample()
    else:
        mean_coe = A_1 * dt + torch.eye(dim)
        mean = torch.einsum('ij,kj->ki', mean_coe, prev) + dt * torch.einsum('ij,kj->ki', B_1, action)  # (m ,dim)
        samples = mean[:, :]
    return samples

def linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact(A, B, sig, running_b,
                                                                       I, m, dt, bd_low_s, bd_upper_s, dim):
    # shape of running_b: (dim, dim)
    # shape of running_c: (dim, 1)
    # shape of output: (m, I, dim)
    # generate data for computing V in 2D case (1st order)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m, dim) + bd_low_s
    dim = A.shape[0]
    exp_Adt = torch.tensor(scipy.linalg.expm(A.numpy() * dt))
    exp_2Adt = torch.tensor(scipy.linalg.expm(A.numpy() * 2 * dt))
    A_1 = 1 / dt * (exp_Adt - torch.eye(dim))
    B_1 = torch.inverse(A) @ A_1 @ B
    C_A = 1 / (2 * dt) * torch.inverse(A) @ (exp_2Adt - torch.eye(dim))

    res = torch.zeros((m, I, dim))
    act_mat = torch.zeros((m, I, dim))
    res[:, 0, :] = init_value[:]  # (m, dim)
    act_mat[:, 0, :] = torch.einsum("ij,kj->ki", running_b, init_value[:])
    prev = res[:, 0, :]  # (m, dim)
    act_temp = act_mat[:, 0, :]  # (m, dim)
    for i in range(1, I):
        # generate and store the i-th sample from previous stored states and actions
        prev = one_step_2D_exact(prev, A_1, B_1, C_A, sig, act_temp, dt, dim)
        res[:, i, :] = prev

        # update and store actions
        act_temp = torch.einsum("ij,kj->ki", running_b, prev)
        act_mat[:, i, :] = act_temp

    return res, act_mat

def linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact_2nd(A, B, sig, running_b,
                                                                       I, m, dt, bd_low_s, bd_upper_s, dim):
    # shape of running_b: (dim, dim)
    # shape of running_c: (dim, 1)
    # shape of output: (m, I, dim)
    # generate data for computing V in 2D case (2nd order)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m, dim) + bd_low_s
    dim = A.shape[0]
    exp_Adt = torch.tensor(scipy.linalg.expm(A.numpy() * dt))
    exp_2Adt = torch.tensor(scipy.linalg.expm(A.numpy() * 2 * dt))
    A_1 = 1 / dt * (exp_Adt - torch.eye(dim))
    B_1 = torch.inverse(A) @ A_1 @ B
    C_A = 1 / (2 * dt) * torch.inverse(A) @ (exp_2Adt - torch.eye(dim))

    res = torch.zeros((m, I, dim))
    act_mat = torch.zeros((m, I, dim))
    res[:, 0, :] = init_value[:]  # (m, dim)
    act_mat[:, 0, :] = torch.einsum("ij,kj->ki", running_b, init_value[:])
    prev = res[:, 0, :]  # (m, dim)
    act_temp = act_mat[:, 0, :]  # (m, dim)
    for i in range(1, I):
        # generate and store the i-th sample from previous stored states and actions
        prev = one_step_2D_exact(prev, A_1, B_1, C_A, sig, act_temp, dt, dim)
        res[:, i, :] = prev

        # update and store actions that will be used for computing the next state in the next iteration
        if i % 2 == 0:
            act_temp = torch.einsum("ij,kj->ki", running_b, prev)
        act_mat[:, i, :] = act_temp

    return res, act_mat

def Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact(A, B, sig, I, m_Q, dt, bd_low_s,
                                            bd_upper_s, bd_low_b, bd_upper_b, dim):
    # res shape: (m_Q, I, dim)
    # generate data for computing Q in 2D case for Phibe (1st order)
    b_temp_tensor = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m_Q, dim, dim)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m_Q, dim) + bd_low_s

    dim = A.shape[0]
    exp_Adt = torch.tensor(scipy.linalg.expm(A.numpy() * dt))
    exp_2Adt = torch.tensor(scipy.linalg.expm(A.numpy() * 2 * dt))
    A_1 = 1 / dt * (exp_Adt - torch.eye(dim))
    B_1 = torch.inverse(A) @ A_1 @ B
    C_A = 1 / (2 * dt) * torch.inverse(A) @ (exp_2Adt - torch.eye(dim))

    res = torch.zeros((m_Q, I, dim))
    res[:, 0, :] = init_value  # (m_Q, dim)
    act_mat = torch.zeros((m_Q, I, dim))
    act_mat[:, 0, :] = (b_temp_tensor.matmul(res[:, 0, :].unsqueeze(-1))).squeeze()
    act_temp = act_mat[:, 0, :]  # (m_Q, dim)

    prev = res[:, 0, :]  # (m_Q, dim)
    for i in range(1, I):
        # generate and store the i-th sample from previous stored states and actions
        prev = one_step_2D_exact(prev, A_1, B_1, C_A, sig, act_temp, dt, dim)
        res[:, i, :] = prev

        # update and store actions
        act_temp = (b_temp_tensor.matmul(res[:, i, :].unsqueeze(-1))).squeeze()
        act_mat[:, i, :] = act_temp

    return res, act_mat

def Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact_2nd(A, B, sig, I, m_Q, dt, bd_low_s,
                                                                  bd_upper_s, bd_low_b, bd_upper_b, dim):
    # res shape: (m_Q, I, dim)
    # generate data for computing Q in 2D case for Phibe (2nd order)
    b_temp_tensor = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m_Q, dim, dim)
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m_Q, dim) + bd_low_s

    dim = A.shape[0]
    exp_Adt = torch.tensor(scipy.linalg.expm(A.numpy() * dt))
    exp_2Adt = torch.tensor(scipy.linalg.expm(A.numpy() * 2 * dt))
    A_1 = 1 / dt * (exp_Adt - torch.eye(dim))
    B_1 = torch.inverse(A) @ A_1 @ B
    C_A = 1 / (2 * dt) * torch.inverse(A) @ (exp_2Adt - torch.eye(dim))

    res = torch.zeros((m_Q, I, dim))
    res[:, 0, :] = init_value  # (m_Q, dim)
    act_mat = torch.zeros((m_Q, I, dim))
    act_mat[:, 0, :] = (b_temp_tensor.matmul(res[:, 0, :].unsqueeze(-1))).squeeze()
    act_temp = act_mat[:, 0, :]  # (m_Q, dim)

    prev = res[:, 0, :]  # (m_Q, dim)
    for i in range(1, I):
        # generate and store the i-th sample from previous stored states and actions
        prev = one_step_2D_exact(prev, A_1, B_1, C_A, sig, act_temp, dt, dim)
        res[:, i, :] = prev

        # update and store actions
        if i % 2 == 0:
            act_temp = (b_temp_tensor.matmul(res[:, i, :].unsqueeze(-1))).squeeze()
        act_mat[:, i, :] = act_temp

    return res, act_mat

def Q_dyn_generator_2D_stochastic_const_act_RL_const_diffusion_exact(A, B, sig, running_b, I, m_Q, dt, bd_low_s,
                                            bd_upper_s, bd_low_a, bd_upper_a, dim):
    # res shape: (m_Q, I, dim)
    # generate data for computing Q in 2D case for RL
    init_value = (bd_upper_s - bd_low_s) * torch.rand(m_Q, dim) + bd_low_s

    dim = A.shape[0]
    exp_Adt = torch.tensor(scipy.linalg.expm(A.numpy() * dt))
    exp_2Adt = torch.tensor(scipy.linalg.expm(A.numpy() * 2 * dt))
    A_1 = 1 / dt * (exp_Adt - torch.eye(dim))
    B_1 = torch.inverse(A) @ A_1 @ B
    C_A = 1 / (2 * dt) * torch.inverse(A) @ (exp_2Adt - torch.eye(dim))

    res = torch.zeros((m_Q, I, dim))
    res[:, 0, :] = init_value # (m_Q, dim)
    act_mat = torch.zeros((m_Q, I, dim))
    act_mat[:, 0, :] = bd_low_a + torch.rand(m_Q, dim) * (bd_upper_a - bd_low_a)
    act_temp = act_mat[:, 0, :] # (m_Q, dim)

    prev = res[:, 0, :] # (m_Q, dim)
    for i in range(1, I):
        # generate and store the i-th sample from previous stored states and actions
        prev = one_step_2D_exact(prev, A_1, B_1, C_A, sig, act_temp, dt, dim)
        res[:, i, :] = prev

        # update and store actions
        act_temp = torch.einsum("ij,kj->ki", running_b, prev)
        act_mat[:, i, :] = act_temp

    return res, act_mat


# 1D merton

def GeoBM_one_step(s_0, mu, sig, dt):
    # s_0 shape (m), return s_dt of shape (m)
    m = s_0.shape[0]
    BM_dt = sqrt(dt) * torch.randn(m)

    return s_0 * torch.exp((mu - (sig**2 / 2)) * dt + sig * BM_dt)


def merton_V_data(r, r_b, mu, sig, running_b, I, m, dt, bd_low_s, bd_upper_s):
    # return shape (m, I) and (m, I), meaning m trajectories and I time steps with actions
    # generate data for computing V
    
    traj_mat = torch.zeros(m, I)
    act_mat = running_b * torch.ones(m, I)

    traj_temp = bd_low_s + (bd_upper_s - bd_low_s) * torch.rand(m)
    traj_mat[:, 0] = traj_temp

    for i in range(1, I):
        if running_b <= 1:
            traj_temp = GeoBM_one_step(traj_temp, (r + (mu - r) * running_b), (sig * running_b), dt)
        else:
            traj_temp = GeoBM_one_step(traj_temp, (r_b + (mu - r_b) * running_b), (sig * running_b), dt)
        traj_mat[:, i] = traj_temp
    return traj_mat, act_mat

def merton_Q_data(r, r_b, mu, sig, I, m, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b):
    # return shape (m, I) and (m, I), meaning m trajectories and I time steps with actions
    # generate data for computing Q
    
    traj_mat = torch.zeros(m, I)
    act_mat = torch.zeros(m, I)

    traj_temp = bd_low_s + (bd_upper_s - bd_low_s) * torch.rand(m)
    act_temp = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m)
    traj_mat[:, 0] = traj_temp
    act_mat[:, 0] = act_temp

    for i in range(1, I):
        # traj_temp = GeoBM_one_step(traj_temp, (r + (mu - r) * act_temp), (sig * act_temp), dt)
        r_mod = torch.where(act_temp > 1, r_b, r)
        traj_temp = GeoBM_one_step(traj_temp, (r_mod + (mu - r_mod) * act_temp), (sig * act_temp), dt)
        traj_mat[:, i] = traj_temp

        # act_temp = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m)
        act_mat[:, i] = act_temp
    
    return traj_mat, act_mat



def merton_Q_data_2nd(r, r_b, mu, sig, I, m, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b):
    # return shape (m, I) and (m, I), meaning m trajectories and I time steps with actions
    # generate data for computing Q
    
    traj_mat = torch.zeros(m, I)
    act_mat = torch.zeros(m, I)

    traj_temp = bd_low_s + (bd_upper_s - bd_low_s) * torch.rand(m)
    act_temp = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m)
    traj_mat[:, 0] = traj_temp
    act_mat[:, 0] = act_temp

    for i in range(1, I):
        r_mod = torch.where(act_temp > 1, r_b, r)
        traj_temp = GeoBM_one_step(traj_temp, (r_mod + (mu - r_mod) * act_temp), (sig * act_temp), dt)
        traj_mat[:, i] = traj_temp

        # if i % 2 == 0:
        #     act_temp = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m)
        act_mat[:, i] = act_temp
    
    return traj_mat, act_mat


def merton_RL_Q_data(r, r_b, mu, sig, running_c, I, m, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b):
    # return shape (m, I) and (m, I), meaning m trajectories and I time steps with actions
    # generate data for computing Q
    
    traj_mat = torch.zeros(m, I)
    act_mat = torch.zeros(m, I)

    traj_temp = bd_low_s + (bd_upper_s - bd_low_s) * torch.rand(m)
    act_temp = bd_low_b + (bd_upper_b - bd_low_b) * torch.rand(m)
    traj_mat[:, 0] = traj_temp
    act_mat[:, 0] = act_temp

    for i in range(1, I):
        # traj_temp = GeoBM_one_step(traj_temp, (r + (mu - r) * act_temp), (sig * act_temp), dt)
        r_mod = torch.where(act_temp > 1, r_b, r)
        traj_temp = GeoBM_one_step(traj_temp, (r_mod + (mu - r_mod) * act_temp), (sig * act_temp), dt)
        traj_mat[:, i] = traj_temp

        act_temp = torch.ones(m) * running_c
        act_mat[:, i] = act_temp
    
    return traj_mat, act_mat