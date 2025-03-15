import torch
from utils import out_put_2d, output_Q

torch.set_default_dtype(torch.float64)


def mat_cal_stochastic(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # computes the matrix for V
    # traj_mat shape (m, I), bases takes (m, I) returns (M+1, m, I),
    # d_bases takes (M+1, m, I) and returns (m, I, M+1), sec_d_bases takes (m, I) and returns (M+1, m, I)
    # return shape (M+1, M+1)

    # Precompute index ranges
    traj_mat_prev = traj_mat[:, :-1]  # (m, I-1)
    traj_mat_next = traj_mat[:, 1:]  # (m, I-1)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)
    gradient = d_bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)
    sec_gradient = sec_d_bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)

    # Compute mu_hat and sig_hat
    mu_hat = (traj_mat_next - traj_mat_prev) / dt  # (m, I-1)
    sig_hat = ((traj_mat_next - traj_mat_prev) ** 2) / dt  # (m, I-1)

    # Compute b
    b = beta * a - mu_hat.unsqueeze(-1) * gradient - 0.5 * sig_hat.unsqueeze(-1) * sec_gradient

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans

def mat_cal_stochastic_deterministic(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # computes the matrix for V
    # traj_mat shape (m, I), bases takes (m, I) returns (M+1, m, I),
    # d_bases takes (M+1, m, I) and returns (m, I, M+1), sec_d_bases takes (m, I) and returns (M+1, m, I)
    # return shape (M+1, M+1)

    # Precompute index ranges
    traj_mat_prev = traj_mat[:, :-1]  # (m, I-1)
    traj_mat_next = traj_mat[:, 1:]  # (m, I-1)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)
    gradient = d_bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)
    sec_gradient = sec_d_bases(traj_mat_prev).permute(1, 2, 0)  # (m, I-1, M+1)

    # Compute mu_hat and sig_hat
    mu_hat = (traj_mat_next - traj_mat_prev) / dt  # (m, I-1)
    sig_hat = torch.zeros_like(mu_hat) # (m, I-1)

    # Compute b
    b = beta * a - mu_hat.unsqueeze(-1) * gradient - 0.5 * sig_hat.unsqueeze(-1) * sec_gradient

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans

def mat_cal_stochastic_2nd_deterministic(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # computes the matrix for V
    # traj_mat shape (m, I), bases takes (m, I) returns (M+1, m, I),
    # d_bases takes (M+1, m, I) and returns (m, I, M+1), sec_d_bases takes (m, I) and returns (M+1, m, I)
    # return shape (M+1, M+1)

    # Precompute index ranges
    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_mat_curr = traj_mat[:, :valid_I:2]  # (m, valid_I)
    traj_mat_next = traj_mat[:, 1:valid_I+1:2]
    traj_mat_next_next = traj_mat[:, 2:valid_I+2:2]  # (m, valid_I)


    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).permute(1, 2, 0)  # (m, valid_I, M+1)
    gradient = d_bases(traj_mat_curr).permute(1, 2, 0)  # (m, valid_I, M+1)

    # Compute mu_hat and sig_hat
    one_step = traj_mat_next - traj_mat_curr
    two_step = traj_mat_next_next - traj_mat_curr

    # mu_hat = (2 * one_step - 0.5 * two_step) / dt  # (m, valid_I)
    mu_hat = (2 * one_step - 0.5 * two_step) / dt  # (m, valid_I)

    # Compute b
    b = beta * a - mu_hat.unsqueeze(-1) * gradient

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans


def mat_cal_stochastic_2nd(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # computes the matrix for V
    # traj_mat shape (m, I), bases takes (m, I) returns (M+1, m, I),
    # d_bases takes (M+1, m, I) and returns (m, I, M+1), sec_d_bases takes (m, I) and returns (M+1, m, I)
    # return shape (M+1, M+1)

    # Precompute index ranges
    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_mat_curr = traj_mat[:, :valid_I:2]  # (m, valid_I)
    traj_mat_next = traj_mat[:, 1:valid_I+1:2]
    traj_mat_next_next = traj_mat[:, 2:valid_I+2:2]  # (m, valid_I)


    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).permute(1, 2, 0)  # (m, valid_I, M+1)
    gradient = d_bases(traj_mat_curr).permute(1, 2, 0)  # (m, valid_I, M+1)
    sec_gradient = sec_d_bases(traj_mat_curr).permute(1, 2, 0)  # (m, valid_I, M+1)

    # Compute mu_hat and sig_hat
    one_step = traj_mat_next - traj_mat_curr
    two_step = traj_mat_next_next - traj_mat_curr

    mu_hat = (2 * one_step - 0.5 * two_step) / dt  # (m, valid_I)
    sig_hat = (2 * (one_step**2) - 0.5 * (two_step**2)) / dt  # (m, valid_I)

    # Compute b
    b = beta * a - mu_hat.unsqueeze(-1) * gradient - 0.5 * sig_hat.unsqueeze(-1) * sec_gradient

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ijl->kl', a, b)

    return ans



def b_cal(traj_mat, reward_mat, bases):
    # traj_mat shape (m, I), reward_mat shape (m, I), bases takes (m, I) returns (M+1, m, I)
    # returns (M+1)

    all_bases = bases(traj_mat[:, :-1]).permute(1, 2, 0)  # Shape should be (m, I-1, M+1)

    weighted_bases = reward_mat[:, :-1].unsqueeze(-1) * all_bases  # Shape will be (m, I-1, M+1)

    ans = weighted_bases.sum(dim=(0, 1))

    return ans

def b_cal_2nd(traj_mat, reward_mat, bases):
    # traj_mat shape (m, I), reward_mat shape (m, I), bases takes (m, I) returns (M+1, m, I)
    # returns (M+1)

    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2
    all_bases = bases(traj_mat[:, :valid_I:2]).permute(1, 2, 0)  # Shape should be (m, I-2, M+1)

    weighted_bases = reward_mat[:, :valid_I:2].unsqueeze(-1) * all_bases  # Shape will be (m, I-2, M+1)

    ans = weighted_bases.sum(dim=(0, 1))

    return ans


def grad_compute_mini_batch(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I) returns (m, I), V_sec_grad takes (m, I) returns (m, I),
    # bases_Q takes(m, I), (m, I) and returns (dim_bases_Q, m, I)
    traj_curr = traj_mat_Q[:, :-1]  # (m, I-1)
    traj_next = traj_mat_Q[:, 1:]  # (m, I-1)
    actions_curr = act_mat_Q[:, :-1]  # (m, I-1)
    reward_mat_crr = reward(traj_curr, actions_curr)  # (m, I-1)

    V_grad_vals_curr = V_grad(traj_curr) # (m, I-1)
    V_sec_grad_curr = V_sec_grad(traj_curr)# (m, I-1)

    diff = traj_next - traj_curr  # (m, I-1)
    diff_sq = diff**2  # (m, I-1)

    mu_hat = (1 / dt) * diff # (m, I-1)
    sig_hat = (1 / (2 * dt)) * diff_sq # (m, I-1)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr + sig_hat * V_sec_grad_curr  # (m, I-1)

    Q_func = out_put_2d(running_coe_Q, bases_Q)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-1)

    term = term_1 - Q_vals_curr  # (m, I-1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I-1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijk,jk->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res

def grad_compute_mini_batch_deterministic(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I) returns (m, I), V_sec_grad takes (m, I) returns (m, I),
    # bases_Q takes(m, I), (m, I) and returns (dim_bases_Q, m, I)
    
    traj_curr = traj_mat_Q[:, :-1]  # (m, I-1)
    traj_next = traj_mat_Q[:, 1:]  # (m, I-1)
    actions_curr = act_mat_Q[:, :-1]  # (m, I-1)
    reward_mat_crr = reward(traj_curr, actions_curr)  # (m, I-1)

    V_grad_vals_curr = V_grad(traj_curr) # (m, I-1)
    # V_sec_grad_curr = V_sec_grad(traj_curr)# (m, I-1)

    diff = traj_next - traj_curr  # (m, I-1)
    # diff_sq = diff**2  # (m, I-1)

    mu_hat = (1 / dt) * diff # (m, I-1)
    # sig_hat = (1 / (2 * dt)) * diff_sq # (m, I-1)

    # term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr + sig_hat * V_sec_grad_curr  # (m, I-1)
    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr  # (m, I-1)

    Q_func = out_put_2d(running_coe_Q, bases_Q)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-1)

    term = term_1 - Q_vals_curr  # (m, I-1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I-1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijk,jk->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res


def grad_compute_mini_batch_2nd(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I) returns (m, I), V_sec_grad takes (m, I) returns (m, I),
    # bases_Q takes(m, I), (m, I) and returns (dim_bases_Q, m, I)
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_curr = traj_mat_Q[:, :valid_I:2]  # (m, valid_I)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2]  # (m, valid_I)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2]  # (m, valid_I)
    actions_curr = act_mat_Q[:, :valid_I:2]  # (m, valid_I)
    reward_mat_crr = reward(traj_curr, actions_curr) #  (m, valid_I)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, valid_I)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, valid_I)

    one_step = traj_next - traj_curr  # (m, valid_I)
    one_sq = one_step**2  # (m, valid_I)

    two_step = traj_next_next - traj_curr  # (m, valid_I)
    two_sq = two_step**2  # (m, valid_I)

    mu_hat = (1 / dt) * (2 * one_step - 0.5 * two_step)  # (m, valid_I)
    sig_hat = (1 / (2 * dt)) * (2 * one_sq - 0.5 * two_sq)  # (m, valid_I)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr + sig_hat * V_sec_grad_curr  # (m, valid_I)

    Q_func = out_put_2d(running_coe_Q, bases_Q)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, valid_I)

    term = term_1 - Q_vals_curr  # (m, valid_I)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, valid_I)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijk,jk->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res

def grad_compute_mini_batch_2nd_deterministic(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I) returns (m, I), V_sec_grad takes (m, I) returns (m, I),
    # bases_Q takes(m, I), (m, I) and returns (dim_bases_Q, m, I)
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_curr = traj_mat_Q[:, :valid_I:2]  # (m, valid_I)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2]  # (m, valid_I)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2]  # (m, valid_I)
    actions_curr = act_mat_Q[:, :valid_I:2]  # (m, valid_I)
    reward_mat_crr = reward(traj_curr, actions_curr) #  (m, valid_I)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, valid_I)
    # V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, valid_I)

    one_step = traj_next - traj_curr  # (m, valid_I)
    # one_sq = one_step**2  # (m, valid_I)

    two_step = traj_next_next - traj_curr  # (m, valid_I)
    # two_sq = two_step**2  # (m, valid_I)

    mu_hat = (1 / dt) * (2 * one_step - 0.5 * two_step)  # (m, valid_I)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr # (m, valid_I)

    Q_func = out_put_2d(running_coe_Q, bases_Q)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, valid_I)

    term = term_1 - Q_vals_curr  # (m, valid_I)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, valid_I)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijk,jk->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res


def galarkin_Q_mat_cal_1st_1d(traj_mat_Q, act_mat_Q, bases_Q):
    traj_curr = traj_mat_Q[:, :-1]  # (m, I-1)
    actions_curr = act_mat_Q[:, :-1]
    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 1)
    mat_Q = torch.einsum("ijk,ljk->il", bases_val, bases_val) # (dim_bases_Q, dim_bases_Q)

    return mat_Q



def galarkin_Q_mat_cal_2nd_1d(traj_mat_Q, act_mat_Q, bases_Q):
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2 
    traj_curr = traj_mat_Q[:, :valid_I:2]  # (m, I-1)
    actions_curr = act_mat_Q[:, :valid_I:2]
    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 2)
    mat_Q = torch.einsum("ijk,ljk->il", bases_val, bases_val) # (dim_bases_Q, dim_bases_Q)

    return mat_Q

def galarkin_Q_b_cal_1st_1d_deterministic(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):

    traj_curr = traj_mat_Q[:, :-1]  # (m, I-1)
    traj_next = traj_mat_Q[:, 1:]  # (m, I-1)
    actions_curr = act_mat_Q[:, :-1]  # (m, I-1)
    reward_mat_crr = reward(traj_curr, actions_curr)  # (m, I-1)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 1)


    V_grad_vals_curr = V_grad(traj_curr) # (m, I-1)

    diff = traj_next - traj_curr  # (m, I-1)

    mu_hat = (1 / dt) * diff # (m, I-1)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr  # (m, I-1)

    b_Q = torch.einsum("ijk,jk->i", bases_val, term_1) # (dim_bases_Q)


    return b_Q


def galarkin_Q_b_cal_2nd_1d_deterministic(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):

    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2

    traj_curr = traj_mat_Q[:, :valid_I:2]  # (m, I-2)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2]  # (m, I-2)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2]  # (m, I-2)
    actions_curr = act_mat_Q[:, :valid_I:2]  # (m, I-2)
    reward_mat_crr = reward(traj_curr, actions_curr) #  (m, I-2)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 2)


    V_grad_vals_curr = V_grad(traj_curr) # (m, I-2)

    one_step = traj_next - traj_curr  # (m, I-2)

    two_step = traj_next_next - traj_curr  # (m, I-2)

    mu_hat = (1 / dt) * (2 * one_step - 0.5 * two_step)  # (m, I-2)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr # (m, I-2)

    b_Q = torch.einsum("ijk,jk->i", bases_val, term_1) # (dim_bases_Q)


    return b_Q


def galarkin_Q_b_cal_1st_1d(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):

    traj_curr = traj_mat_Q[:, :-1]  # (m, I-1)
    traj_next = traj_mat_Q[:, 1:]  # (m, I-1)
    actions_curr = act_mat_Q[:, :-1]  # (m, I-1)
    reward_mat_crr = reward(traj_curr, actions_curr)  # (m, I-1)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 1)


    V_grad_vals_curr = V_grad(traj_curr) # (m, I-1)
    V_sec_grad_curr = V_sec_grad(traj_curr)# (m, I-1)

    diff = traj_next - traj_curr  # (m, I-1)
    diff_sq = diff**2  # (m, I-1)

    mu_hat = (1 / dt) * diff # (m, I-1)
    sig_hat = (1 / (2 * dt)) * diff_sq # (m, I-1)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr + sig_hat * V_sec_grad_curr  # (m, I-1)

    b_Q = torch.einsum("ijk,jk->i", bases_val, term_1) # (dim_bases_Q)


    return b_Q


def galarkin_Q_b_cal_2nd_1d(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):

    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2

    traj_curr = traj_mat_Q[:, :valid_I:2]  # (m, I-2)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2]  # (m, I-2)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2]  # (m, I-2)
    actions_curr = act_mat_Q[:, :valid_I:2]  # (m, I-2)
    reward_mat_crr = reward(traj_curr, actions_curr) #  (m, I-2)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I - 1)


    V_grad_vals_curr = V_grad(traj_curr) # (m, I-1)
    V_sec_grad_curr = V_sec_grad(traj_curr)# (m, I-1)

    one_step = traj_next - traj_curr  # (m, I-2)
    one_sq = one_step**2  # (m, I-2)

    two_step = traj_next_next - traj_curr  # (m, I-2)
    two_sq = two_step**2  # (m, I-2)

    mu_hat = (1 / dt) * (2 * one_step - 0.5 * two_step)  # (m, I-2)
    sig_hat = (1 / (2 * dt)) * (2 * one_sq - 0.5 * two_sq)  # (m, I-2)

    term_1 = reward_mat_crr + mu_hat * V_grad_vals_curr + sig_hat * V_sec_grad_curr  # (m, I-2)

    b_Q = torch.einsum("ijk,jk->i", bases_val, term_1) # (dim_bases_Q)


    return b_Q


def mat_cal_stochastic_2D(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # bases: takes (m, I, dim) returns (M+1, m, I, 1)
    # d_bases: takes (m, I, dim) returns (M+1, m, I, dim)
    # sec_d_bases takes (m, I, dim) returns (M + 1, m, I, dim, dim)
    # output shape: (M+1, M+1)


    # Precompute index ranges
    traj_mat_curr = traj_mat[:, :-1, :]  # (m, I-1, dim)
    traj_mat_next = traj_mat[:, 1:, :]  # (m, I-1, dim)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).squeeze()  # (M+1, m, I-1)
    gradient = d_bases(traj_mat_curr)  # (M+1, m, I-1, dim)
    sec_gradient = sec_d_bases(traj_mat_curr)  # (M + 1, m, I - 1, dim, dim)

    # Compute mu_hat and sig_hat
    diff = traj_mat_next - traj_mat_curr  # (m, I-1, dim)
    mu_hat = diff / dt  # (m, I-1, dim)
    sig_hat = torch.einsum("ijk,ijl->ijkl", diff, diff) / dt  # (m, i-1, dim, dim)


    # Compute b
    b = beta * a - torch.einsum('ijkl,jkl->ijk', gradient, mu_hat) - 0.5 * torch.einsum("ijklm,jklm->ijk", sec_gradient, sig_hat) # (M+1, m, I-1)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans

def mat_cal_deterministic_2D(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # bases: takes (m, I, dim) returns (M+1, m, I, 1)
    # d_bases: takes (m, I, dim) returns (M+1, m, I, dim)
    # sec_d_bases takes (m, I, dim) returns (M + 1, m, I, dim, dim)
    # output shape: (M+1, M+1)


    # Precompute index ranges
    traj_mat_curr = traj_mat[:, :-1, :]  # (m, I-1, dim)
    traj_mat_next = traj_mat[:, 1:, :]  # (m, I-1, dim)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).squeeze()  # (M+1, m, I-1)
    gradient = d_bases(traj_mat_curr)  # (M+1, m, I-1, dim)
    # sec_gradient = sec_d_bases(traj_mat_curr)  # (M + 1, m, I - 1, dim, dim)

    # Compute mu_hat and sig_hat
    diff = traj_mat_next - traj_mat_curr  # (m, I-1, dim)
    mu_hat = diff / dt  # (m, I-1, dim)
    # sig_hat = torch.einsum("ijk,ijl->ijkl", diff, diff) / dt  # (m, i-1, dim, dim)


    # Compute b
    b = beta * a - torch.einsum('ijkl,jkl->ijk', gradient, mu_hat) # (M+1, m, I-1)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans

def mat_cal_stochastic_2D_2nd(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # bases: takes (m, I, dim) returns (M+1, m, I, 1)
    # d_bases: takes (m, I, dim) returns (M+1, m, I, dim)
    # sec_d_bases takes (m, I, dim) returns (M + 1, m, I, dim, dim)
    # output shape: (M+1, M+1)


    # Precompute index ranges
    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_mat_curr = traj_mat[:, :valid_I:2, :]  # (m, valid_I, dim)
    traj_mat_next = traj_mat[:, 1:valid_I+1:2, :]  # (m, valid_I, dim)
    traj_mat_next_next = traj_mat[:, 2:valid_I+2:2, :]  # (m, valid_I, dim)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).squeeze()  # (M+1, m, valid_I)
    gradient = d_bases(traj_mat_curr) # (M+1, m, valid_I, dim)
    sec_gradient = sec_d_bases(traj_mat_curr) # (M + 1, m, valid_I, dim, dim)

    # Compute mu_hat and sig_hat
    diff, sec_diff = traj_mat_next - traj_mat_curr, traj_mat_next_next - traj_mat_curr  # (m, valid_I, dim), (m, valid_I, dim)
    mu_hat = (2 * diff - 0.5 * sec_diff) / dt   # (m, valid_I, dim)
    sig_hat = (2 * torch.einsum("ijk,ijl->ijkl", diff, diff) -
               0.5 * torch.einsum("ijk,ijl->ijkl", sec_diff, sec_diff)) / dt  # (m, valid_I, dim, dim)


    # Compute b
    b = beta * a - torch.einsum('ijkl,jkl->ijk', gradient, mu_hat) - 0.5 * torch.einsum("ijklm,jklm->ijk", sec_gradient, sig_hat)
    # (M+1, m, valid_I)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans


def mat_cal_deterministic_2D_2nd(traj_mat, bases, d_bases, sec_d_bases, dt, beta):
    # M+1 is the number of the bases
    # traj_mat: (m, I, dim)
    # bases: takes (m, I, dim) returns (M+1, m, I, 1)
    # d_bases: takes (m, I, dim) returns (M+1, m, I, dim)
    # sec_d_bases takes (m, I, dim) returns (M + 1, m, I, dim, dim)
    # output shape: (M+1, M+1)


    # Precompute index ranges
    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_mat_curr = traj_mat[:, :valid_I:2, :]  # (m, valid_I, dim)
    traj_mat_next = traj_mat[:, 1:valid_I+1:2, :]  # (m, valid_I, dim)
    traj_mat_next_next = traj_mat[:, 2:valid_I+2:2, :]  # (m, valid_I, dim)

    # Compute a, gradient, sec_gradient
    a = bases(traj_mat_curr).squeeze()  # (M+1, m, valid_I)
    gradient = d_bases(traj_mat_curr) # (M+1, m, valid_I, dim)
    # sec_gradient = sec_d_bases(traj_mat_curr) # (M + 1, m, valid_I, dim, dim)

    # Compute mu_hat and sig_hat
    diff, sec_diff = traj_mat_next - traj_mat_curr, traj_mat_next_next - traj_mat_curr  # (m, valid_I, dim), (m, valid_I, dim)
    mu_hat = (2 * diff - 0.5 * sec_diff) / dt   # (m, I-2, dim)


    # Compute b
    b = beta * a - torch.einsum('ijkl,jkl->ijk', gradient, mu_hat)
    # (M+1, m, valid_I)

    # Compute outer products and accumulate
    ans = torch.einsum('ijk,ljk->il', a, b)

    return ans


def b_cal_2D(traj_mat, reward_mat, bases):
    # traj_mat shape: (m, I, dim), reward_mat shape: (m, I, 1), bases takes in (m, I, dim) and outputs (M+1, m, I, 1)
    # output shape: (M+1)

    all_bases = bases(traj_mat[:,:-1,:]) # Shape should be (M+1, m, I-1, 1)
    ans = torch.einsum("ijkl,jkl->i", all_bases, reward_mat[:,:-1,:])

    return ans


def b_cal_2D_2nd(traj_mat, reward_mat, bases):
    # traj_mat shape: (m, I, dim), reward_mat shape: (m, I, 1), bases takes in (m, I, dim) and outputs (M+1, m, I, 1)
    # output shape: (M+1)
    I = traj_mat.shape[1]
    valid_I = (I // 2) * 2 - 2

    all_bases = bases(traj_mat[:,:valid_I:2,:]) # Shape should be (M+1, m, valid_I, 1)
    ans = torch.einsum("ijkl,jkl->i", all_bases, reward_mat[:,:valid_I:2,:])

    return ans


def grad_compute_2D_mini(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)
    # m_Q_GD = int(bs / I)
    traj_curr = traj_mat_Q[:, :-1, :]  # (m, I-1, dim)
    traj_next = traj_mat_Q[:, 1:, :]  # (m, I-1, dim)
    actions_curr = act_mat_Q[:, :-1,: ]  # (m, I-1, dim)
    reward_mat_curr = reward(traj_curr, actions_curr) # (m, I-1, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-1, dim)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-1, dim, dim)

    diff = traj_next - traj_curr  # (m, I-1, dim)
    diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff) # (m, I-1, dim, dim)

    mu_hat = (1 / dt) * diff  # (m, I-1, dim)
    sig_hat = (1 / dt) * diff_sq  # (m, I-1, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1) + 0.5 * torch.einsum("ijkl,ijkl->ij", sig_hat, V_sec_grad_curr).unsqueeze(-1)  # (m, I-1, 1)

    Q_func = output_Q(running_coe_Q, bases_Q)  # function that takes (m, I, dim), (m, I, dim) and returns (m, I, 1)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-1, 1)

    term = term_1 - Q_vals_curr  # (m, I-1, 1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, I-1, 1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijkl,jkl->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res

def grad_compute_2D_mini_deterministic(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)
    # m_Q_GD = int(bs / I)
    traj_curr = traj_mat_Q[:, :-1, :]  # (m, I-1, dim)
    traj_next = traj_mat_Q[:, 1:, :]  # (m, I-1, dim)
    actions_curr = act_mat_Q[:, :-1,: ]  # (m, I-1, dim)
    reward_mat_curr = reward(traj_curr, actions_curr) # (m, I-1, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-1, dim)
    # V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-1, dim, dim)

    diff = traj_next - traj_curr  # (m, I-1, dim)
    # diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff) # (m, I-1, dim, dim)

    mu_hat = (1 / dt) * diff  # (m, I-1, dim)
    # sig_hat = (1 / dt) * diff_sq  # (m, I-1, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1)  # (m, I-1, 1)

    Q_func = output_Q(running_coe_Q, bases_Q)  # function that takes (m, I, dim), (m, I, dim) and returns (m, I, 1)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-1, 1)

    term = term_1 - Q_vals_curr  # (m, I-1, 1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, I-1, 1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijkl,jkl->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res


def grad_compute_2D_mini_2nd(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)
    # m_Q_GD = int(bs / I)
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2

    traj_curr = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    traj_next = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    traj_next_next = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    actions_curr = act_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    reward_mat_curr = reward(traj_curr, actions_curr)  # (m, I-2, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-2, dim)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-2, dim, dim)

    diff = traj_next - traj_curr  # (m, I-2, dim)
    diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff)  # (m, I-2, dim, dim)

    sec_diff = traj_next_next - traj_curr  # (m, I-2, dim)
    sec_diff_sq = torch.einsum("ijk,ijl->ijkl", sec_diff, sec_diff)  # (m, I-2, dim, dim)

    mu_hat = (1 / dt) * (2 * diff - 0.5 * sec_diff)  # (m, I-2, dim)
    sig_hat = (1 / dt) * (2 * diff_sq - 0.5 * sec_diff_sq)  # (m, I-2, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1) \
             + 0.5 * torch.einsum("ijkl,ijkl->ij", sig_hat, V_sec_grad_curr).unsqueeze(-1)  # (m, I-2, 1)

    Q_func = output_Q(running_coe_Q, bases_Q)  # function that takes (m, I, dim), (m, I, dim) and returns (m, I, 1)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-2, 1)

    term = term_1 - Q_vals_curr  # (m, I-2, 1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, I-2, 1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijkl,jkl->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res


def grad_compute_2D_mini_2nd_deterministic(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    # compute the gradient
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)
    # m_Q_GD = int(bs / I)
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2

    traj_curr = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    traj_next = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    traj_next_next = traj_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    actions_curr = act_mat_Q[:, :valid_I:2, :]  # (m, I-2, dim)
    reward_mat_curr = reward(traj_curr, actions_curr)  # (m, I-2, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-2, dim)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-2, dim, dim)

    diff = traj_next - traj_curr  # (m, I-2, dim)
    # diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff)  # (m, I-2, dim, dim)

    sec_diff = traj_next_next - traj_curr  # (m, I-2, dim)
    # sec_diff_sq = torch.einsum("ijk,ijl->ijkl", sec_diff, sec_diff)  # (m, I-2, dim, dim)

    mu_hat = (1 / dt) * (2 * diff - 0.5 * sec_diff)  # (m, I-2, dim)
    # sig_hat = (1 / dt) * (2 * diff_sq - 0.5 * sec_diff_sq)  # (m, I-2, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1)  # (m, I-2, 1)

    Q_func = output_Q(running_coe_Q, bases_Q)  # function that takes (m, I, dim), (m, I, dim) and returns (m, I, 1)

    Q_vals_curr = Q_func(traj_curr, actions_curr)  # (m, I-2, 1)

    term = term_1 - Q_vals_curr  # (m, I-2, 1)

    grad_Q_curr = bases_Q(traj_curr, actions_curr)  # (dim_bases_Q, m, I-2, 1)

    res = - (1 / (traj_curr.shape[0] * traj_curr.shape[1])) * torch.einsum("ijkl,jkl->i", grad_Q_curr, term)
    # (dim_bases_Q)

    return res

def galarkin_Q_mat_cal_1st_2d(traj_mat_Q, act_mat_Q, bases_Q):
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)
    traj_curr = traj_mat_Q[:, :-1, :]  # (m, I-1, dim)
    actions_curr = act_mat_Q[:, :-1,: ]  # (m, I-1, dim)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I-1, 1)

    mat_Q = torch.einsum('ijkp,ljkp->il', bases_val, bases_val)

    return mat_Q



def galarkin_Q_mat_cal_2nd_2d(traj_mat_Q, act_mat_Q, bases_Q):
    # running_coe_Q shape (dim_base_Q), V_grad takes (m, I, dim) returns (m, I, dim),
    # V_sec_grad takes (m, I, dim) returns (m, I, dim, dim),
    # bases_Q takes (m, I, dim), (m, I, dim) and returns (dim_bases_Q, m, I, 1),
    # reward takes (m, I, dim), (m, I, dim) and returns (m, I, 1)

    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2

    traj_curr = traj_mat_Q[:, :valid_I:2, :]  # (m, valid_I, dim)
    actions_curr = act_mat_Q[:, :valid_I:2,: ]  # (m, valid_I, dim)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, valid_I, 1)

    mat_Q = torch.einsum('ijkp,ljkp->il', bases_val, bases_val)

    return mat_Q


def galarkin_Q_b_cal_1st_2d(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    traj_curr = traj_mat_Q[:, :-1, :]  # (m, I-1, dim)
    traj_next = traj_mat_Q[:, 1:, :]  # (m, I-1, dim)
    actions_curr = act_mat_Q[:, :-1,: ]  # (m, I-1, dim)
    reward_mat_curr = reward(traj_curr, actions_curr) # (m, I-1, 1)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I-1, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-1, dim)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-1, dim, dim)

    diff = traj_next - traj_curr  # (m, I-1, dim)
    diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff) # (m, I-1, dim, dim)

    mu_hat = (1 / dt) * diff  # (m, I-1, dim)
    sig_hat = (1 / dt) * diff_sq  # (m, I-1, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1) + 0.5 * torch.einsum("ijkl,ijkl->ij", sig_hat, V_sec_grad_curr).unsqueeze(-1)  # (m, I-1, 1)



    b_Q = torch.einsum("ijkl,jkl->i", bases_val, term_1)
    # (dim_bases_Q)

    return b_Q

def galarkin_Q_b_cal_1st_2d_deterministic(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    traj_curr = traj_mat_Q[:, :-1, :]  # (m, I-1, dim)
    traj_next = traj_mat_Q[:, 1:, :]  # (m, I-1, dim)
    actions_curr = act_mat_Q[:, :-1,: ]  # (m, I-1, dim)
    reward_mat_curr = reward(traj_curr, actions_curr) # (m, I-1, 1)

    bases_val = bases_Q(traj_curr, actions_curr) # (dim_bases_Q, m, I-1, 1)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-1, dim)
    # V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-1, dim, dim)

    diff = traj_next - traj_curr  # (m, I-1, dim)
    # diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff) # (m, I-1, dim, dim)

    mu_hat = (1 / dt) * diff  # (m, I-1, dim)
    # sig_hat = (1 / dt) * diff_sq  # (m, I-1, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1)   # (m, I-1, 1)



    b_Q = torch.einsum("ijkl,jkl->i", bases_val, term_1)
    # (dim_bases_Q)

    return b_Q



def galarkin_Q_b_cal_2nd_2d(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):
    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_curr = traj_mat_Q[:, :valid_I:2, :]  # (m, valid_I, dim)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2, :]  # (m, valid_I, dim)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2, :]  # (m, valid_I, dim)
    actions_curr = act_mat_Q[:, :valid_I:2,: ]  # (m, valid_I, dim)
    reward_mat_curr = reward(traj_curr, actions_curr)  # (m, valid_I, 1)

    bases_val = bases_Q(traj_curr, actions_curr)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, I-2, dim)
    V_sec_grad_curr = V_sec_grad(traj_curr)  # (m, I-2, dim, dim)

    diff = traj_next - traj_curr  # (m, I-2, dim)
    diff_sq = torch.einsum("ijk,ijl->ijkl", diff, diff)  # (m, I-2, dim, dim)

    sec_diff = traj_next_next - traj_curr  # (m, I-2, dim)
    sec_diff_sq = torch.einsum("ijk,ijl->ijkl", sec_diff, sec_diff)  # (m, I-2, dim, dim)

    mu_hat = (1 / dt) * (2 * diff - 0.5 * sec_diff)  # (m, I-2, dim)
    sig_hat = (1 / dt) * (2 * diff_sq - 0.5 * sec_diff_sq)  # (m, I-2, dim, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1) \
             + 0.5 * torch.einsum("ijkl,ijkl->ij", sig_hat, V_sec_grad_curr).unsqueeze(-1)  # (m, I-2, 1)



    b_Q = torch.einsum("ijkl,jkl->i", bases_val, term_1)
    # (dim_bases_Q)

    return b_Q


def galarkin_Q_b_cal_2nd_2d_deterministic(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt):

    I = traj_mat_Q.shape[1]
    valid_I = (I // 2) * 2 - 2
    traj_curr = traj_mat_Q[:, :valid_I:2, :]  # (m, valid_I, dim)
    traj_next = traj_mat_Q[:, 1:valid_I+1:2, :]  # (m, valid_I, dim)
    traj_next_next = traj_mat_Q[:, 2:valid_I+2:2, :]  # (m, valid_I, dim)
    actions_curr = act_mat_Q[:, :valid_I:2,: ]  # (m, valid_I, dim)
    reward_mat_curr = reward(traj_curr, actions_curr)  # (m, valid_I, 1)

    bases_val = bases_Q(traj_curr, actions_curr)

    V_grad_vals_curr = V_grad(traj_curr)  # (m, valid_I, dim)

    diff = traj_next - traj_curr  # (m, valid_I, dim)

    sec_diff = traj_next_next - traj_curr  # (m, valid_I, dim)

    mu_hat = (1 / dt) * (2 * diff - 0.5 * sec_diff)  # (m, valid_I, dim)

    term_1 = reward_mat_curr + torch.einsum("ijk,ijk->ij", mu_hat, V_grad_vals_curr).unsqueeze(-1)  # (m, valid_I, 1)



    b_Q = torch.einsum("ijkl,jkl->i", bases_val, term_1)
    # (dim_bases_Q)

    return b_Q