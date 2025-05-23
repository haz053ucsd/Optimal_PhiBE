# Standard libraries
from typing import Callable, List, Tuple, Dict, Union

# Third-party libraries
import torch
from tqdm import tqdm

# Project-specific modules
from utils import (
    output_V, out_put, l_2_compute_1D_V, true_V_eval_1D, true_V_eval_2D, l_2_compute_2D_V, merton_policy_eval, dist_compute_merton
)
from data_generator import (
    Q_dyn_generator_const_act_exact, Q_dyn_generator_const_act_2nd_exact, linear_dyn_generator_stochastic_const_act_exact, linear_dyn_generator_stochastic_const_act_exact_2nd,Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact,
    linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact, linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact_2nd, Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact_2nd, merton_V_data, merton_Q_data, merton_Q_data_2nd
)
from phibe_utils import (
    mat_cal_stochastic, mat_cal_stochastic_2nd, b_cal, b_cal_2nd,
    grad_compute_mini_batch, grad_compute_mini_batch_2nd, mat_cal_stochastic_2D, b_cal_2D, grad_compute_2D_mini, galarkin_Q_b_cal_1st_2d,
galarkin_Q_b_cal_1st_1d, galarkin_Q_b_cal_2nd_1d, galarkin_Q_b_cal_2nd_2d, galarkin_Q_mat_cal_1st_1d, galarkin_Q_mat_cal_1st_2d, galarkin_Q_mat_cal_2nd_1d,
grad_compute_2D_mini_2nd, galarkin_Q_mat_cal_2nd_2d, mat_cal_stochastic_2D_2nd, b_cal_2D_2nd, mat_cal_stochastic_deterministic, galarkin_Q_b_cal_1st_1d_deterministic,
    mat_cal_stochastic_2nd_deterministic, galarkin_Q_b_cal_2nd_1d_deterministic, grad_compute_mini_batch_deterministic, grad_compute_mini_batch_2nd_deterministic,
mat_cal_deterministic_2D, mat_cal_deterministic_2D_2nd, galarkin_Q_b_cal_1st_2d_deterministic, galarkin_Q_b_cal_2nd_2d_deterministic,
grad_compute_2D_mini_deterministic, grad_compute_2D_mini_2nd_deterministic)

torch.set_default_dtype(torch.float64)

def phibe_finder_1D_LQR(
    beta: float,
    b_init: float,
    Q_init: torch.Tensor,
    bd_low_s: float,
    bd_upper_s: float,
    bd_low_b: float,
    bd_upper_b: float,
    reward: Callable,
    simple_basis: bool,
    bases_V: Callable,
    d_bases_V: Callable,
    sec_d_bases_V: Callable,
    bases_Q: Callable,
    num_iter: int,
    deterministic: bool,
    Q_method: str,
    GD_num_iter: int,
    m: int,
    m_Q: int,
    I: int,
    lr: float,
    dt: float,
    order: int,
    true_V: torch.Tensor,
    info_true: Dict[str, Union[float, torch.Tensor]],
) -> Tuple[List[float], List[float], Callable]:
    """
        Optimal Phibe for the 1D LQR problem. 

        Args:
            beta (float): Discount factor.
            b_init (float): Initial value of parameter b.
            Q_init (torch.Tensor): Initial coefficient matrix for Q if use gradient descent.
            bd_low_s, bd_upper_s (float): Lower and upper bounds for state space for generating trajectory data.
            bd_low_b, bd_upper_b, bd_low_c, bd_upper_c (float): Bounds for policy parameters b and c
            for generating trajectory data.
            reward (Callable): Reward function.
            simple_basis (bool): If True, then the basis functions do not include 1, and if False, 1 is included.
            bases_V, d_bases_V, sec_d_bases_V, bases_Q (Callable): Basis functions for value function and Q-function.
            num_iter (int): Number of main iterations.
            deterministic (bool): whether use deterministic PhiBE (treat diffusion as constant). In the paper, it was proved that,
            for LQR, deterministic option is better no matter what the dynamics are.
            Q_method (str): Method used for Q evaluation "GD" or "Galerkin", "Galerkin" is better and faster.
            GD_num_iter (int): Number of gradient descent iterations for Q.
            m (int): Number of trajectories for V evaluation.
            m_Q (int): Number of trajectories for Q gradient descent.
            I (int): Number of time steps when generating trajectories.
            lr (float): Learning rate for Q updates.
            dt (float): Time step size.
            order (int): Order of the Phibe (1 or 2).
            true_V (torch.Tensor): True value function.
            info_true (Dict): Contains true LQR parameters for generating trajectories.

        Returns:
            Tuple containing:
                - List of b values,
                - List of distances between optimal and current value functions at each iteration.
                - Callable value function of the returned optimal policy
        """

    if Q_method != "GD" and Q_method != "Galerkin":
        raise ValueError("Invalid method. Supported methods are GD or Galerkin.")
    if Q_method == "GD":
        if (simple_basis and Q_init.shape[0] != 3) or (not simple_basis and Q_init.shape[0] != 4):
            raise ValueError("Invalid Initialization of Q coefficient.")
    if true_V.shape[0] != 3:
        raise ValueError("True optimal value function should be the coefficients under basis 1, s, s^2.")

    # Initialization
    running_b, running_coe_Q = b_init, Q_init
    b_val, V_lst, V_exact_dist = [], [], []

    if deterministic:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_deterministic, b_cal, \
                grad_compute_mini_batch_deterministic, galarkin_Q_mat_cal_1st_1d, \
                galarkin_Q_b_cal_1st_1d_deterministic
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2nd_deterministic, b_cal_2nd, \
                grad_compute_mini_batch_2nd_deterministic, \
                galarkin_Q_mat_cal_2nd_1d, galarkin_Q_b_cal_2nd_1d_deterministic
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")
    else:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic, b_cal, grad_compute_mini_batch, galarkin_Q_mat_cal_1st_1d, \
                galarkin_Q_b_cal_1st_1d
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2nd, b_cal_2nd, grad_compute_mini_batch_2nd, \
                galarkin_Q_mat_cal_2nd_1d, galarkin_Q_b_cal_2nd_1d
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")

    # Generate data for Q evaluation
    if order == 1:
        traj_mat_Q, act_mat_Q = Q_dyn_generator_const_act_exact(info_true["A"], info_true["B"], info_true["sig"], I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b,)  # (m, I)
    elif order == 2:
        traj_mat_Q, act_mat_Q = Q_dyn_generator_const_act_2nd_exact(info_true["A"], info_true["B"], info_true["sig"], I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b)  # (m, I)

    for _ in tqdm(range(num_iter), desc=f"Running Optimal Phibe of order {order} using {Q_method}"):
        # Record b and c, collect statistics
        b_val.append(running_b)
        V_pi = true_V_eval_1D(info_true["A"], info_true["B"], info_true["sig"], info_true["Q"], info_true["R"], beta, running_b)
        V_lst.append(V_pi)
        V_exact_dist.append(l_2_compute_1D_V(true_V - V_pi, 3, -3))


        if order == 2:
            traj_mat, act_mat = linear_dyn_generator_stochastic_const_act_exact_2nd(info_true["A"], info_true["B"], info_true["sig"], running_b, I, m, dt, bd_low_s, bd_upper_s)
        elif order == 1:
            traj_mat, act_mat = linear_dyn_generator_stochastic_const_act_exact(info_true["A"], info_true["B"], info_true["sig"], running_b, I, m, dt, bd_low_s, bd_upper_s)
        reward_mat = reward(traj_mat, act_mat)

        # Policy evaluation
        mat_V, b_V = mat_V_comp(traj_mat, bases_V, d_bases_V, sec_d_bases_V, dt, beta), b_V_comp(traj_mat, reward_mat, bases_V)
        coe_V = torch.inverse(mat_V).matmul(b_V)
        V_grad, V_sec_grad = out_put(coe_V, d_bases_V), out_put(coe_V, sec_d_bases_V)

        # Update for Q
        if Q_method == "GD":
            index = 0
            while index < GD_num_iter:
                Q_grad = grad_Q_comp(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt)  # (dim_bases_Q)
                running_coe_Q -= lr * Q_grad  # (dim_bases_Q)
                index += 1
        elif Q_method == "Galerkin":
            mat_Q = mat_cal_Q(traj_mat_Q, act_mat_Q, bases_Q)
            b_Q = b_cal_Q(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt)
            running_coe_Q = torch.inverse(mat_Q) @ b_Q


        if not simple_basis:
            running_b = - running_coe_Q[2] / (2 * running_coe_Q[1])
        else:
            running_b = - running_coe_Q[1] / (2 * running_coe_Q[0])

    return b_val, V_exact_dist, V_pi


def phibe_finder_2D_LQR(
        beta: float,
        b_init: torch.Tensor,
        Q_init: torch.Tensor,
        bd_low_s: float,
        bd_upper_s: float,
        bd_low_b: float,
        bd_upper_b: float,
        reward: Callable,
        simp_basis: bool,
        bases_V: Callable,
        d_bases_V: Callable,
        sec_d_bases_V: Callable,
        bases_Q: Callable,
        num_iter: int,
        deterministic: bool,
        Q_method: str,
        GD_num_iter: int,
        m: int,
        m_Q: int,
        I: int,
        lr: float,
        dt: float,
        order: int,
        true_V: torch.Tensor,
        info_true: Dict[str, Union[torch.Tensor, float]],
) -> Tuple[List[torch.Tensor], List[float], torch.Tensor]:
    """
    Optimal Phibe for the 2D LQR problem.

    Args:
        beta (float): Discount factor.
        b_init (torch.Tensor): Initial parameter b (2D tensor of shape (2, 2)).
        Q_init (torch.Tensor): Initial coefficient for Q (1D tensor of shape (15)).
        bd_low_s, bd_upper_s (float): Lower and upper bounds for state space for trajectory data generation.
        bd_low_b, bd_upper_b, bd_low_c, bd_upper_c (float): Bounds for b and c for trajectory generation.
        reward (Callable): Reward function.
        simple_basis (bool): If True, then the basis functions do not include 1, and if False, 1 is included.
        bases_V, d_bases_V, sec_d_bases_V, bases_Q (Callable): Basis functions for value function and Q-function.
        num_iter (int): Number of main iterations.
        deterministic (bool): whether use deterministic PhiBE (treat diffusion as constant). In the paper, it was proved that,
        for LQR, deterministic option is better no matter what the dynamics are.
        Q_method (str): Method used for Q evaluation "GD" or "Galerkin"
        GD_num_iter (int): Number of gradient descent iterations for Q.
        m (int): Number of trajectories for V evaluation.
        m_Q (int): Number of trajectories for Q gradient descent.
        I (int): Number of time steps when generating trajectories.
        lr (float): Learning rate for Q updates.
        dt (float): Time step size.
        order (int): Order of the Phibe (1 or 2).
        true_V (torch.Tensor): True value function.
        info_true (Dict): Contains true LQR parameters for generating trajectories.

    Returns:
        Tuple containing:
            - List of b values,
            - List of distances between optimal and current value functions at each iteration.
            - Callable value function of the returned optimal policy
    """
    if Q_method != "GD" and Q_method != "Galerkin":
        raise ValueError("Invalid method. Supported methods are GD or Galerkin.")
    if not deterministic:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2D, b_cal_2D, grad_compute_2D_mini, galarkin_Q_mat_cal_1st_2d, \
                galarkin_Q_b_cal_1st_2d
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2D_2nd, b_cal_2D_2nd, grad_compute_2D_mini_2nd, \
                galarkin_Q_mat_cal_2nd_2d, galarkin_Q_b_cal_2nd_2d
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")
    elif deterministic:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_deterministic_2D, b_cal_2D, grad_compute_2D_mini_deterministic, \
                galarkin_Q_mat_cal_1st_2d, galarkin_Q_b_cal_1st_2d_deterministic
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_deterministic_2D_2nd, b_cal_2D_2nd, \
                grad_compute_2D_mini_2nd_deterministic, galarkin_Q_mat_cal_2nd_2d, galarkin_Q_b_cal_2nd_2d_deterministic
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")


    running_b, running_coe_Q = b_init[:, :], Q_init[:]
    b_val, V_lst, V_exact_dist = [], [], []
    A, B, sig, R, Q = info_true["A"], info_true["B"], info_true["sig"], info_true["R"], info_true["Q"]

    # Generate data for Q evaluation
    if order == 1:
        traj_mat_Q, act_mat_Q = Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact(A, B, sig, I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b, dim=2)
    elif order == 2:
        traj_mat_Q, act_mat_Q = Q_dyn_generator_2D_stochastic_const_act_const_diffusion_exact_2nd(A, B, sig, I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b, dim=2)

    for _ in tqdm(range(num_iter), desc=f"Running Optimal Phibe of order {order} using {Q_method}"):
        b_val.append(running_b)
        if order == 1:
            traj_mat, act_mat = linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact(A, B, sig, running_b, I, m, dt, bd_low_s, bd_upper_s, dim=2)
        elif order == 2:
            traj_mat, act_mat = linear_dyn_generator_stochastic_2D_const_act_const_diffusion_exact_2nd(A, B, sig, running_b, I, m, dt, bd_low_s, bd_upper_s, dim=2)

        # returns (m, I, dim), (m, I, dim)
        reward_mat = reward(traj_mat, act_mat)  # returns (m, I, 1)

        # policy evaluation
        mat_V = mat_V_comp(traj_mat, bases_V, d_bases_V, sec_d_bases_V, dt, beta) # returns (M+1)
        b_V = b_V_comp(traj_mat, reward_mat, bases_V)
        coe_V = torch.inverse(mat_V).matmul(b_V)
        V_grad = output_V(coe_V, d_bases_V)
        V_sec_grad = output_V(coe_V, sec_d_bases_V)

        # collect statistics
        V_pi = true_V_eval_2D(A, B, running_b, R, Q, beta, sig)
        V_lst.append(V_pi)
        if not V_pi is None:
            V_exact_dist.append(l_2_compute_2D_V(V_pi - true_V, 3, -3, 3, -3))
        else:
            V_exact_dist.append(torch.nan)

        # update for Q
        if Q_method == "GD":
            index = 0
            while index < GD_num_iter:
                Q_grad = grad_Q_comp(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt)
                running_coe_Q -= lr * Q_grad  # (dim_bases_Q)
                index += 1
        elif Q_method == "Galerkin":
            mat_Q = mat_cal_Q(traj_mat_Q, act_mat_Q, bases_Q)
            b_Q = b_cal_Q(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt)
            running_coe_Q = torch.inverse(mat_Q) @ b_Q

        correction = 1 if simp_basis else 0
        
        N_mat = torch.tensor([[running_coe_Q[2 - correction], running_coe_Q[3 - correction]],
                              [running_coe_Q[4 - correction], running_coe_Q[5 - correction]]])
        L_mat = torch.tensor(
            [[running_coe_Q[9 - correction], 0.5 * running_coe_Q[6 - correction]],
             [0.5 * running_coe_Q[6 - correction], running_coe_Q[-1]]])
        running_b = - 0.5 * torch.inverse(L_mat).matmul(N_mat.T)

    return b_val, V_exact_dist, V_pi



def phibe_finder_1D_merton(
    beta: float,
    b_init: float,
    Q_init: torch.Tensor,
    bd_low_s: float,
    bd_upper_s: float,
    bd_low_b: float,
    bd_upper_b: float,
    reward: Callable,
    bases_V: Callable,
    d_bases_V: Callable,
    sec_d_bases_V: Callable,
    bases_Q: Callable,
    num_iter: int,
    deterministic: bool,
    Q_method: str,
    GD_num_iter: int,
    m: int,
    m_Q: int,
    I: int,
    lr: float,
    dt: float,
    order: int,
    true_V: torch.Tensor,
    info_true: Dict[str, Union[float, torch.Tensor]],
) -> Tuple[List[float]]:
    """
        Optimal Phibe for the 1D LQR problem. (need modification for this description)

        Args:
            beta (float): Discount factor.
            b_init (float): Initial value of parameter b.
            Q_init (torch.Tensor): Initial coefficient matrix for Q.
            bd_low_s, bd_upper_s (float): Lower and upper bounds for state space for generating trajectory data.
            bd_low_b, bd_upper_b, bd_low_c, bd_upper_c (float): Bounds for policy parameters b and c
            for generating trajectory data.
            reward (Callable): Reward function.
            bases_V, d_bases_V, sec_d_bases_V, bases_Q (Callable): Basis functions for value function and Q-function.
            num_iter (int): Number of main iterations.
            deterministic (bool): whether use deterministic PhiBE (treat diffusion as constant).
            Q_method (str): Method used for Q evaluation "GD" or "Galerkin"
            GD_num_iter (int): Number of gradient descent iterations for Q.
            m (int): Number of trajectories for V evaluation.
            m_Q (int): Number of trajectories for Q gradient descent.
            I (int): Number of time steps when generating trajectories.
            lr (float): Learning rate for Q updates.
            dt (float): Time step size.
            order (int): Order of the Phibe (1 or 2).
            true_V (torch.Tensor): True value function.
            info_true (Dict): Contains true LQR parameters for generating trajectories.

        Returns:
            Tuple containing:
                - List of b values,
                - List of distances between optimal and current value functions at each iteration.
                - Callable value function of the returned optimal policy
        """

    if Q_method != "GD" and Q_method != "Galerkin":
        raise ValueError("Invalid method. Supported methods are GD or Galerkin.")

    # Initialization
    running_b, running_coe_Q = b_init, Q_init
    b_val, l2_dist = [], []

    if deterministic:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_deterministic, b_cal, \
                grad_compute_mini_batch_deterministic, galarkin_Q_mat_cal_1st_1d, \
                galarkin_Q_b_cal_1st_1d_deterministic
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2nd_deterministic, b_cal_2nd, \
                grad_compute_mini_batch_2nd_deterministic, \
                galarkin_Q_mat_cal_2nd_1d, galarkin_Q_b_cal_2nd_1d_deterministic
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")
    else:
        if order == 1:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic, b_cal, grad_compute_mini_batch, galarkin_Q_mat_cal_1st_1d, \
                galarkin_Q_b_cal_1st_1d
        elif order == 2:
            mat_V_comp, b_V_comp, grad_Q_comp, mat_cal_Q, b_cal_Q = mat_cal_stochastic_2nd, b_cal_2nd, grad_compute_mini_batch_2nd, \
                galarkin_Q_mat_cal_2nd_1d, galarkin_Q_b_cal_2nd_1d
        else:
            raise ValueError("Invalid order. Supported values are 1 or 2.")

    # Generate data for Q evaluation
    if order == 1:
        traj_mat_Q, act_mat_Q = merton_Q_data(info_true['r'], info_true['r_b'], info_true['mu'], info_true['sig'], I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b)  # (m, I)
    elif order == 2:
        traj_mat_Q, act_mat_Q = merton_Q_data_2nd(info_true['r'], info_true['r_b'], info_true['mu'], info_true['sig'], I, m_Q, dt, bd_low_s, bd_upper_s, bd_low_b, bd_upper_b) # (m, I)

    for _ in tqdm(range(num_iter), desc=f"Running Optimal Phibe of order {order} using {Q_method}"):
        # Record b, collect statistics
        b_val.append(running_b)
        V_pi = merton_policy_eval(info_true['mu'], info_true['r'], info_true['r_b'], info_true['sig'], info_true['gamma'], running_b, beta)
        l2_dist.append(dist_compute_merton(true_V - V_pi))

        if order == 2:
            traj_mat, act_mat = merton_V_data(info_true['r'], info_true['r_b'], info_true['mu'], info_true['sig'], running_b, I, m, dt, bd_low_s, bd_upper_s)
        elif order == 1:
            traj_mat, act_mat = merton_V_data(info_true['r'], info_true['r_b'], info_true['mu'], info_true['sig'], running_b, I, m, dt, bd_low_s, bd_upper_s)
        reward_mat = reward(traj_mat, act_mat)

        # Policy evaluation
        mat_V, b_V = mat_V_comp(traj_mat, bases_V, d_bases_V, sec_d_bases_V, dt, beta), b_V_comp(traj_mat, reward_mat, bases_V)
        coe_V = torch.inverse(mat_V).matmul(b_V)
        V_grad, V_sec_grad = out_put(coe_V, d_bases_V), out_put(coe_V, sec_d_bases_V)

        # Update for Q
        if Q_method == "GD":
            index = 0
            while index < GD_num_iter:
                Q_grad = grad_Q_comp(traj_mat_Q, act_mat_Q, running_coe_Q, V_grad, V_sec_grad, bases_Q, reward, dt)  # (dim_bases_Q)
                running_coe_Q -= lr * Q_grad  # (dim_bases_Q)
                index += 1
        elif Q_method == "Galerkin":
            mat_Q = mat_cal_Q(traj_mat_Q, act_mat_Q, bases_Q)
            b_Q = b_cal_Q(traj_mat_Q, act_mat_Q, V_grad, V_sec_grad, bases_Q, reward, dt)
            running_coe_Q = torch.inverse(mat_Q) @ b_Q
        if - 0.5 * running_coe_Q[1] / running_coe_Q[2] <= 0:
            running_b = b_init
            continue
        else:
            running_b = - 0.5 * running_coe_Q[1] / running_coe_Q[2]
        
    V_func = output_V(coe_V, bases_V)

    return b_val, l2_dist, V_func