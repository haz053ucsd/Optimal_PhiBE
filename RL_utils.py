import torch

torch.set_default_dtype(torch.float64)

def mat_Q_cal_stochastic_RL(traj_mat, act_mat, bases_Q, dt, beta):
    # computes the matrix for Q
    # traj_mat shape (m, I), act_mat shape (m, I), bases_Q takes (m, I), (m, I) returns (dim_bases_Q, m, I)
    #return shape (M+1, M+1)

    # Precompute index ranges
    traj_mat_prev = traj_mat[:, :-1]  # (m, I-1)
    act_mat_prev = act_mat[:, :-1] # (m, I-1)
    traj_mat_next = traj_mat[:, 1:]  # (m, I-1)
    act_mat_next = act_mat[:, 1:]

    a = bases_Q(traj_mat_prev, act_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)

    # Compute b
    b = a - torch.exp(- torch.tensor([beta * dt])) * bases_Q(traj_mat_next, act_mat_next).permute(1, 2, 0)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans


def mat_V_cal_stochastic_RL(traj_mat, bases_V, dt, beta):
    # computes the matrix for Q
    # traj_mat shape (m, I), act_mat shape (m, I), bases_Q takes (m, I), (m, I) returns (dim_bases_Q, m, I)
    #return shape (M+1, M+1)

    # Precompute index ranges
    traj_mat_prev = traj_mat[:, :-1]  # (m, I-1)
    traj_mat_next = traj_mat[:, 1:]  # (m, I-1)

    # Compute a, gradient, sec_gradient
    a = bases_V(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)

    # Compute b
    b = a - torch.exp(- torch.tensor([beta * dt])) * bases_V(traj_mat_next).permute(1, 2, 0)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans


def b_cal_Q_RL(traj_mat, act_mat, reward_mat, bases_Q, dt):
    # traj_mat shape (m, I), reward_mat shape (m, I), bases takes (m, I), (m, I) returns (M+1, m, I)
    # returns (M+1)

    all_bases = bases_Q(traj_mat[:,:-1], act_mat[:,:-1]).permute(1, 2, 0)  # Shape should be (m, I-1, M+1)

    weighted_bases = reward_mat[:,:-1].unsqueeze(-1) * all_bases  # Shape will be (m, I-1, M+1)

    ans = weighted_bases.sum(dim=(0, 1)) * dt

    return ans


def b_cal_V_RL(traj_mat, reward_mat, bases_V, dt):
    # traj_mat shape (m, I), reward_mat shape (m, I), bases takes (m, I), (m, I) returns (M+1, m, I)
    # returns (M+1)

    all_bases = bases_V(traj_mat[:,:-1]).permute(1, 2, 0)  # Shape should be (m, I-1, M+1)

    weighted_bases = reward_mat[:,:-1].unsqueeze(-1) * all_bases  # Shape will be (m, I-1, M+1)

    ans = weighted_bases.sum(dim=(0, 1)) * dt

    return ans


def mat_Q_cal_stochastic_RL_2D(traj_mat, act_mat, bases_Q, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # act_mat: (m, I, dim)
    # bases_Q: takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1)
    # output shape: (M+1, M+1)

    # Precompute index ranges
    traj_mat_curr = traj_mat[:, :-1, :]  # (m, I-1, dim)
    traj_mat_next = traj_mat[:, 1:, :]  # (m, I-1, dim)
    act_mat_curr = act_mat[:, :-1, :]  # (m, I-1, dim)
    act_mat_next = act_mat[:, 1:, :]  # (m, I-1, dim)

    # Compute a, b
    a = bases_Q(traj_mat_curr, act_mat_curr).squeeze()  # (M+1, m, I-1)
    b = a - torch.exp(- torch.tensor([beta * dt])) * bases_Q(traj_mat_next, act_mat_next).squeeze() # (M+1, m, I-1)
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans


def b_cal_Q_2D_RL(traj_mat, act_mat, reward_mat, bases_Q, dt):
    # traj_mat shape: (m, I, dim), act_mat shape: (m, I, dim), reward_mat shape: (m, I, 1),
    # bases_Q: takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1)
    # output shape: (M+1)

    all_bases = bases_Q(traj_mat[:, :-1, :], act_mat[:, :-1, :])  # Shape should be (M+1, m, I-1, 1)
    ans = torch.einsum("ijkl,jkl->i", all_bases, reward_mat[:, :-1, :]) * dt

    return ans


def mat_V_cal_stochastic_RL_2D(traj_mat, bases, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # bases: takes in (m, I, dim) and outputs (M+1, m, I, 1)
    # output shape: (M+1, M+1)

    # Precompute index ranges
    traj_mat_curr = traj_mat[:, :-1, :]  # (m, I-1, dim)
    traj_mat_next = traj_mat[:, 1:, :]  # (m, I-1, dim)

    # Compute a, b
    a = bases(traj_mat_curr).squeeze()  # (M+1, m, I-1)
    b = a - torch.exp(- torch.tensor([beta * dt])) * bases(traj_mat_next).squeeze()  # (M+1, m, I-1)
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans


def b_cal_V_2D_RL(traj_mat, reward_mat, bases, dt):
    # traj_mat shape: (m, I, dim), reward_mat shape: (m, I, 1),
    # bases: takes in (m, I, dim) and outputs (M+1, m, I, 1)
    # output shape: (M+1)

    all_bases = bases(traj_mat[:, :-1, :])  # Shape should be (M+1, m, I-1, 1)
    ans = torch.einsum("ijkl,jkl->i", all_bases, reward_mat[:, :-1, :]) * dt

    return ans
