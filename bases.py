import torch

torch.set_default_dtype(torch.float64)

# bases functions for 1D LQR

def bases_poly(s):
    # 1, s^2
    return torch.stack([torch.ones_like(s), s**2])

def bases_poly_simp(s):
    # s^2
    return torch.stack([s**2])

def d_bases_poly(s):
    # 1, s^2
    return torch.stack([torch.zeros_like(s), 2 * s])

def d_bases_poly_simp(s):
    # s^2
    return torch.stack([2 * s])

def sec_bases_poly(s):
    # 1, s^2
    return torch.stack([torch.zeros_like(s), 2 * torch.ones_like(s)])

def sec_bases_poly_simp(s):
    # s**2
    return torch.stack([2 * torch.ones_like(s)])

def bases_2d(s, a):
    # 1, a^2, sa, s^2
    return torch.stack([torch.ones_like(s), a**2, s * a, s**2])

def bases_2d_simp(s, a):
    # a^2, sa, s^2
    return torch.stack([a ** 2, s * a, s ** 2])

# bases functions for 2D LQR

def bases_poly_2D(mat):
    # mat shape: (m, I, dim)
    # order: (s_1, s_2) = (0,0), (0,2), (1,1), (2,0)
    # output shape: (4, m, I, 1)

    ans_00 = torch.ones(mat.shape[0], mat.shape[1], 1, device=mat.device)
    ans_02 = mat[:, :, 1:2] ** 2  # Square the second dimension of mat
    ans_11 = mat[:, :, 0:1] * mat[:, :, 1:2]  # Multiply first and second dimensions
    ans_20 = mat[:, :, 0:1] ** 2  # Square the first dimension of mat

    # Stack all results along 0 dimension (M+1 dimension)
    # ans = torch.cat([ans_00, ans_01, ans_02, ans_10, ans_11, ans_20], dim=-1).permute(3, 0, 1, 2)
    ans = torch.stack([ans_00, ans_02, ans_11, ans_20], dim=0)
    return ans

def bases_poly_2D_simp(mat):
    # mat shape: (m, I, dim)
    # order: (s_1, s_2) = (0,2), (1,1), (2,0)
    # output shape: (3, m, I, 1)

    ans_02 = mat[:, :, 1:2] ** 2  # Square the second dimension of mat
    ans_11 = mat[:, :, 0:1] * mat[:, :, 1:2]  # Multiply first and second dimensions
    ans_20 = mat[:, :, 0:1] ** 2  # Square the first dimension of mat

    # Stack all results along 0 dimension (M+1 dimension)
    # ans = torch.cat([ans_00, ans_01, ans_02, ans_10, ans_11, ans_20], dim=-1).permute(3, 0, 1, 2)
    ans = torch.stack([ans_02, ans_11, ans_20], dim=0)
    return ans

def d_bases_poly_2D(mat):
    # mat shape: (m, I-1, dim)
    # order: (s_1, s_2) = (0,0), (0,2), (1,1), (2,0)
    # output shape: (4, m, I-1, 2), in the last dimension, the first is the partial derivative with respective to s_1, the second is with respective to s_2

    # Compute each polynomial term directly
    ans_00 = torch.zeros(mat.shape[0], mat.shape[1], 2, device=mat.device)
    ans_02 = torch.cat((torch.zeros(mat.shape[0], mat.shape[1], 1, device=mat.device), 2 * mat[:, :, 1:2]), dim=-1)
    ans_11 = torch.cat((mat[:, :, 1:2], mat[:, :, 0:1]), dim=-1)
    ans_20 = torch.cat((2 * mat[:, :, 0:1], torch.zeros(mat.shape[0], mat.shape[1], 1, device=mat.device)), dim=-1)

    # Stack all results along 0 dim
    ans = torch.stack([ans_00, ans_02, ans_11, ans_20], dim=0)
    return ans

def d_bases_poly_2D_simp(mat):
    # mat shape: (m, I-1, dim)
    # order: (s_1, s_2) = (0,2), (1,1), (2,0)
    # output shape: (3, m, I-1, 2), in the last dimension, the first is the partial derivative with respective to s_1, the second is with respective to s_2

    # Compute each polynomial term directly
    ans_02 = torch.cat((torch.zeros(mat.shape[0], mat.shape[1], 1, device=mat.device), 2 * mat[:, :, 1:2]), dim=-1)
    ans_11 = torch.cat((mat[:, :, 1:2], mat[:, :, 0:1]), dim=-1)
    ans_20 = torch.cat((2 * mat[:, :, 0:1], torch.zeros(mat.shape[0], mat.shape[1], 1, device=mat.device)), dim=-1)

    # Stack all results along 0 dim
    ans = torch.stack([ans_02, ans_11, ans_20], dim=0)
    return ans

def sec_d_bases_poly_2D(mat):
    # mat shape: (m, I, dim)
    # order of the first dimension: (s_1, s_2) = (0,0), (0,2), (1,1), (2,0)
    # order of the last two dimension: 0 means partial derivative w.r.t. s_1, 1 means partial derivative w.r.t. s_2. It will be symmetric.
    # return shape (4, m, I, dim, dim)
    ans_00 = torch.zeros(mat.shape[0], mat.shape[1], 2, 2, device=mat.device) # (m, I, 2, 2)
    mat_02 = torch.tensor([[0., 0.], [0., 2.]])
    ans_02 = mat_02.expand(mat.shape[0], mat.shape[1], 2, 2)
    mat_11 = torch.eye(2)
    ans_11 = mat_11.expand(mat.shape[0], mat.shape[1], 2, 2)
    mat_20 = torch.tensor([[2., 0.], [0., 0.]])
    ans_20 = mat_20.expand(mat.shape[0], mat.shape[1], 2, 2)

    ans = torch.stack([ans_00, ans_02, ans_11, ans_20], dim=0)

    return ans

def sec_d_bases_poly_2D_simp(mat):
    # mat shape: (m, I, dim)
    # order of the first dimension: (s_1, s_2) = (0,2), (1,1), (2,0)
    # order of the last two dimension: 0 means partial derivative w.r.t. s_1, 1 means partial derivative w.r.t. s_2. It will be symmetric.
    # return shape (3, m, I, dim, dim)
    mat_02 = torch.tensor([[0., 0.], [0., 2.]])
    ans_02 = mat_02.expand(mat.shape[0], mat.shape[1], 2, 2)
    mat_11 = torch.eye(2).float()
    ans_11 = mat_11.expand(mat.shape[0], mat.shape[1], 2, 2)
    mat_20 = torch.tensor([[2., 0.], [0., 0.]])
    ans_20 = mat_20.expand(mat.shape[0], mat.shape[1], 2, 2)

    ans = torch.stack([ans_02, ans_11, ans_20], dim=0)

    return ans

def bases_poly_2D_s_a(traj_mat, act_mat):
    # traj_mat shape: (m, I, dim)
    # act_mat shape: (m, I, dim)
    # order 1, s1s2, s1a1, s1a2, s2a1, s2a2, a1a2, si^2, s2^2, a1^2, a2^2
    # output shape: (11, m, I, 1)

    ans_00 = torch.ones(traj_mat.shape[0], traj_mat.shape[1], 1, device=traj_mat.device) # (m, I, 1)
    ans_s1 = traj_mat[:, :, 0:1]
    ans_s2 = traj_mat[:, :, 1:2]
    ans_a1 = act_mat[:, :, 0:1]
    ans_a2 = act_mat[:, :, 1:2]
    ans_s1s2 =  ans_s1 * ans_s2
    ans_s1a1 = ans_s1 * ans_a1
    ans_s1a2 = ans_s1 * ans_a2
    ans_s2a1 = ans_s2 * ans_a1
    ans_s2a2 = ans_s2 * ans_a2
    ans_a1a2 = ans_a1 * ans_a2
    ans_s1_sq = ans_s1**2
    ans_s2_sq = ans_s2**2
    ans_a1_sq = ans_a1**2
    ans_a2_sq = ans_a2**2

    # Stack all results along 0 dimension (11 dimension)

    ans = torch.stack([ans_00, ans_s1s2, ans_s1a1, ans_s1a2, ans_s2a1, ans_s2a2, ans_a1a2, ans_s1_sq, ans_s2_sq, ans_a1_sq, ans_a2_sq], dim=0)
    return ans

def bases_poly_2D_s_a_simp(traj_mat, act_mat):
    # traj_mat shape: (m, I, dim)
    # act_mat shape: (m, I, dim)
    # order s1s2, s1a1, s1a2, s2a1, s2a2, a1a2, si^2, s2^2, a1^2, a2^2
    # output shape: (10, m, I, 1)

    ans_s1 = traj_mat[:, :, 0:1]
    ans_s2 = traj_mat[:, :, 1:2]
    ans_a1 = act_mat[:, :, 0:1]
    ans_a2 = act_mat[:, :, 1:2]
    ans_s1s2 = ans_s1 * ans_s2
    ans_s1a1 = ans_s1 * ans_a1
    ans_s1a2 = ans_s1 * ans_a2
    ans_s2a1 = ans_s2 * ans_a1
    ans_s2a2 = ans_s2 * ans_a2
    ans_a1a2 = ans_a1 * ans_a2
    ans_s1_sq = ans_s1 ** 2
    ans_s2_sq = ans_s2 ** 2
    ans_a1_sq = ans_a1 ** 2
    ans_a2_sq = ans_a2 ** 2

    # Stack all results along 0 dimension (10 dimension)

    ans = torch.stack(
        [ans_s1s2, ans_s1a1, ans_s1a2, ans_s2a1, ans_s2a2, ans_a1a2, ans_s1_sq, ans_s2_sq, ans_a1_sq,
         ans_a2_sq], dim=0)
    return ans