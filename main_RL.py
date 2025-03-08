import torch
from tqdm import tqdm
from typing import Callable, List, Tuple, Dict, Union

# Project-specific modules
from utils import (
    output_Q, output_V, out_put, true_V_eval_1D, true_V_eval_2D, l_2_compute_1D_V, LQR_1D_true_solution, l_2_compute_2D_V, LQR_2D_true_solution
)
from data_generator import (
    linear_dyn_generator_stochastic_Q_exact, linear_dyn_generator_stochastic_const_act_exact,
Q_dyn_generator_2D_stochastic_const_act_RL_const_diffusion_exact, linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact

)

from RL_utils import (
    mat_Q_cal_stochastic_RL, b_cal_Q_RL, mat_Q_cal_stochastic_RL_2D, b_cal_Q_2D_RL,
    mat_V_cal_stochastic_RL, mat_V_cal_stochastic_RL_2D, b_cal_V_2D_RL, b_cal_V_RL
)
torch.set_default_dtype(torch.float64)


def RL_finder_1D_LQR(
    beta: float,
    b_init: float,
    bd_low_s: float,
    bd_upper_s: float,
    bd_low_a: float,
    bd_upper_a: float,
    reward: Callable,
    simple_basis: bool,
    bases_Q: Callable,
    num_iter: int,
    m_Q: int,
    I: int,
    dt: float,
    true_V: torch.Tensor,
    info_true: Dict[str, Union[float, torch.Tensor]],
) -> Tuple[List[float], List[float]]:
    """
    RL method for the 1D LQR problem.

    Args:
        beta (float): Discount factor.
        b_init (float): Initial value of parameter b.
        c_init (float): Initial value of parameter c.
        bd_low_s, bd_upper_s (float): Lower and upper bounds for state space for generating trajectory data.
        bd_low_a, bd_upper_a (float): Bounds for actions a for generating trajectory data.
        reward (Callable): Reward function.
        bases_Q (Callable): Basis functions for Q-function.
        num_iter (int): Number of main iterations.
        m (int): Number of trajectories for V evaluation at the end.
        m_Q (int): Number of trajectories for Q evaluation.
        I (int): Number of time steps when generating trajectories.
        dt (float): Time step size.
        true_V (torch.Tensor): True value function.
        info_true (Dict): Contains true LQR parameters for generating trajectories.

    Returns:
        Tuple containing:
            - List of b values,
            - List of distances between optimal and current value functions at each iteration.
    """
    # Initialization
    running_b = b_init
    b_val, V_exact_dist = [], []

    for _ in tqdm(range(num_iter), desc=f"Running RL method"):
        # Record b and c and collect statistics
        b_val.append(running_b)
        V_pi = true_V_eval_1D(info_true["A"], info_true["B"], info_true["sig"], info_true["Q"], info_true["R"],  beta, running_b)
        V_exact_dist.append(l_2_compute_1D_V(true_V - V_pi, 3, -3))

        # Generate trajectories for policy evaluation
        traj_mat, act_mat = linear_dyn_generator_stochastic_Q_exact(info_true["A"], info_true["B"], info_true["sig"], running_b, I, int(m_Q), dt, bd_low_s, bd_upper_s, bd_low_a, bd_upper_a)
        reward_mat = reward(traj_mat, act_mat)

        mat_Q = mat_Q_cal_stochastic_RL(traj_mat, act_mat, bases_Q, dt, beta)
        b_Q = b_cal_Q_RL(traj_mat, act_mat, reward_mat, bases_Q, dt)
        running_coe_Q = torch.inverse(mat_Q).matmul(b_Q)

        if not simple_basis:
            running_b = - running_coe_Q[2] / (2 * running_coe_Q[1])
        else:
            running_b = - running_coe_Q[1] / (2 * running_coe_Q[0])


    return b_val, V_exact_dist


def RL_finder_2D_LQR(
        beta: float,
        b_init: torch.Tensor,
        bd_low_s: float,
        bd_upper_s: float,
        bd_low_a: float,
        bd_upper_a: float,
        reward: Callable,
        simple_basis: bool,
        bases_Q: Callable,
        num_iter: int,
        I: int,
        m_Q: int,
        dt: float,
        true_V: torch.Tensor,
        info_true: Dict[str, Union[torch.Tensor, float]],
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    RL method for the 2D LQR problem.

    Args:
        beta (float): Discount factor.
        b_init (torch.Tensor): Initial parameter b (2D tensor of shape (2, 2)).
        c_init (torch.Tensor): Initial parameter c (2D tensor of shape (2, 1)).
        Q_init (torch.Tensor): Initial coefficient for Q (1D tensor of shape (15)).
        bd_low_s, bd_upper_s (float): Lower and upper bounds for state space for trajectory data generation.
        bd_low_a, bd_upper_a: Bounds for a for trajectory generation.
        reward (Callable): Reward function.
        bases_Q (Callable): Basis functions for Q-function.
        bases_V (Callable): Basis functions for V-function.
        num_iter (int): Number of main iterations.
        I (int): Number of time steps
        m (int): Number of trajectories for V evaluation.
        m_Q (int): Number of trajectories for Q evaluation.
        dt: (float): Time step size.
        true_V (torch.Tensor): True value function.
        info_true (Dict): Contains true LQR parameters for generating trajectories.

    Returns:
        Tuple containing:
            - List of b values (torch.Tensors),
            - List of c values (torch.Tensors),
            - List of distances between true and estimated value functions at each iteration,
            - coefficient of the approximated V
    """

    running_b= b_init[:, :]
    b_val, V_exact_dist = [], []
    A, B, sig, R, Q = info_true["A"], info_true["B"], info_true["sig"], info_true["R"], info_true["Q"]

    for _ in tqdm(range(num_iter), desc=f"Running RL method"):
        b_val.append(running_b)
        V_pi = true_V_eval_2D(A, B, running_b, R, Q, beta, sig)
        if not V_pi is None:
            V_exact_dist.append(l_2_compute_2D_V(V_pi - true_V, 3, -3, 3, -3))
        else:
            V_exact_dist.append(torch.nan)

        traj_mat, act_mat = Q_dyn_generator_2D_stochastic_const_act_RL_const_diffusion_exact(A, B, sig, running_b, I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_a, bd_upper_a, dim=2)
        # returns (m, I, dim), (m, I, dim)
        reward_mat = reward(traj_mat, act_mat)  # returns (m, I, 1)

        # update for Q
        mat_Q = mat_Q_cal_stochastic_RL_2D(traj_mat, act_mat, bases_Q, dt, beta)
        b_Q = b_cal_Q_2D_RL(traj_mat, act_mat, reward_mat, bases_Q, dt)
        running_coe_Q = torch.inverse(mat_Q) @ b_Q


        # update b and c by solving the optimization problem
        # (in the LQR case, it amounts to optimize a quadratic function)
        # correction = 1 if not const_basis else 0
        #
        # K_vec = torch.tensor([[running_coe_Q[3 - correction]], [running_coe_Q[4 - correction]]])
        # N_mat = torch.tensor([[running_coe_Q[6 - correction], running_coe_Q[7 - correction]],
        #                       [running_coe_Q[8 - correction], running_coe_Q[9 - correction]]])
        # L_mat = torch.tensor(
        #     [[running_coe_Q[13 - correction], 0.5 * running_coe_Q[10 - correction]],
        #      [0.5 * running_coe_Q[10 - correction], running_coe_Q[-1]]])
        # running_b = - 0.5 * torch.inverse(L_mat).matmul(N_mat.T)
        # running_c = torch.inverse(L_mat).matmul(K_vec)
        correction = 1 if simple_basis else 0
        N_mat = torch.tensor([[running_coe_Q[2 - correction], running_coe_Q[3 - correction]],
                              [running_coe_Q[4 - correction], running_coe_Q[5 - correction]]])
        L_mat = torch.tensor(
            [[running_coe_Q[9 - correction], 0.5 * running_coe_Q[6 - correction]],
             [0.5 * running_coe_Q[6 - correction], running_coe_Q[-1]]])
        running_b = - 0.5 * torch.inverse(L_mat).matmul(N_mat.T)



    return b_val, V_exact_dist