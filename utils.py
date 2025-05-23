import torch
from tqdm import tqdm
import sympy as sp
from math import sqrt
import numpy as np
from scipy.linalg import solve_continuous_are, expm

device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

def out_put(theta, bases):
    def out(s):
        bases_val = bases(s).permute(1, 2, 0) # (m, I-1, M+1)
        ans = torch.einsum("k,ijk->ij", theta, bases_val)
        return ans
    return out


def out_put_2d(theta, bases):
    def out(s, a):
        bases_val = bases(s, a).permute(1, 2, 0) # (m, I-1, M+1)
        ans = torch.einsum("k,ijk->ij", theta, bases_val)
        return ans
    return out


def output_V(coe, bases):
    # coe shape (dim_bases)
    # bases takes (m, I, dim) and output (dim_bases, m, I, ..)
    # return a function that takes (m, I, dim) and outputs (m, I, ...)
    def ans(mat):
        # mat shape (m, I, dim)
        bases_val = bases(mat)
        res = torch.einsum("i,ij...->j...", coe, bases_val)
        return res
    return ans


def output_Q(coe, bases):
    # coe shape (dim_bases)
    # bases takes (m, I, dim), (m, I, dim) and output (dim_bases, m, I, 1)
    # return a function that takes (m, I, dim), (m, I, dim) and outputs (m, I, 1)
    def ans(mat_1, mat_2):
        # mat_1 shape (m, I, dim), mat_2 shape (m, I, dim)
        bases_val = bases(mat_1, mat_2)  # (dim_bases, m, I, 1)
        res = torch.einsum("i,ij...->j...", coe, bases_val)
        return res
    return ans


# l2 distance calculator
def l_2_compute_1D_V(coe_V, upper, lower):
    x = sp.symbols("x")
    p_x = sum(c * (x**i) for i, c in enumerate(coe_V))
    p_x_sq = p_x**2

    integral = sp.integrate(p_x_sq, (x, lower, upper))

    return sp.sqrt(integral)


def l_2_compute_1D_Q(coe_Q, upper_s, lower_s, upper_a, lower_a):
    # coe_Q order: 1, a, a^2, s, sa, s^2
    s, a = sp.symbols("s a")
    Q_func = coe_Q[0] + coe_Q[1] * a + coe_Q[2] * (a**2) + coe_Q[3] * s + coe_Q[4] * s * a + coe_Q[5] * (s**2)

    Q_func_sq = Q_func**2

    integral = sp.integrate(Q_func_sq, (s, lower_s, upper_s), (a, lower_a, upper_a))

    return float(sp.sqrt(integral))

def l_2_compute_2D_V(coe_V, upper_s1, lower_s1, upper_s2, lower_s2):
    # coe_V order: order: 1, s_2**2, s_1s_2, s_1**2
    s1, s2 = sp.symbols("s1 s2")
    Q_func = coe_V[0] + coe_V[1] * (s2**2) + coe_V[2] * s1 * s2 + coe_V[3] * (s1**2)

    Q_func_sq = Q_func**2

    integral = sp.integrate(Q_func_sq, (s1, lower_s1, upper_s1), (s2, lower_s2, upper_s2))

    return float(sp.sqrt(integral))

# true solution finder
def LQR_1D_true_solution(A, B, Q, R, sigma, beta):
    # returns the coefficient of the policy under basis (s), and of V under basis (1, s, s^2)
    a = B**2 / R
    b = 2 * A - beta
    c = -Q

    # Solve the quadratic equation: a * P^2 + b * P + c = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("Discriminant is negative. No real solutions for P.")

    # Compute the roots
    P1 = (-b + np.sqrt(discriminant)) / (2 * a)
    P2 = (-b - np.sqrt(discriminant)) / (2 * a)

    P = P1 if P1 < 0 else P2

    # Compute C
    if beta == 0:
        C = 0
    else:
        C = (sigma**2 * P) / beta

    return P * B / R, torch.tensor([C , 0., P])

def LQR_1D_PhiBE_true_solution_1st_order(A, B, Q, R, sigma, beta, dt):
    # returns the coefficient of the policy under basis (s), and of V under basis (1, s, s^2) using the PhiBE
    A_hat = (np.exp(A * dt) - 1) / dt
    B_hat = 1 / A * A_hat * B
    a = B_hat**2 / R
    b =  (2 * A_hat - beta)
    c = -Q

    # Solve the quadratic equation: a * P^2 + b * P + c = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("Discriminant is negative. No real solutions for P.")

    # Compute the roots
    P1 = (-b + np.sqrt(discriminant)) / (2 * a)
    P2 = (-b - np.sqrt(discriminant)) / (2 * a)

    P = P1 if P1 < 0 else P2

    # Compute C
    if beta == 0:
        C = 0
    else:
        C = (sigma**2 * P) / beta

    return P * B_hat / R, torch.tensor([C , 0., P])


def LQR_1D_PhiBE_true_solution_2nd_order(A, B, Q, R, sigma, beta, dt):
    # returns the coefficient of the policy under basis (s), and of V under basis (1, s, s^2) using the PhiBE
    A_hat = (2 * (np.exp(A * dt) - 1) - 0.5 * (np.exp(2 * A * dt) - 1)) / dt
    B_hat = 1 / A * A_hat * B
    a = B_hat**2 / R
    b = 2 * A_hat - beta
    c = -Q

    # Solve the quadratic equation: a * P^2 + b * P + c = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("Discriminant is negative. No real solutions for P.")

    # Compute the roots
    P1 = (-b + np.sqrt(discriminant)) / (2 * a)
    P2 = (-b - np.sqrt(discriminant)) / (2 * a)

    P = P1 if P1 < 0 else P2

    # Compute C
    if beta == 0:
        C = 0
    else:
        C = (sigma**2 * P) / beta

    return P * B_hat / R, torch.tensor([C , 0., P])


def LQR_1D_BE_true_solution(A, B, Q, R, sigma, beta, dt):
    # returns the coefficient of the policy under basis (s), and of V under basis (1, s, s^2) using the PhiBE
    A_hat = (np.exp(A * dt) - 1) / dt
    B_hat = 1 / A * A_hat * B
    
    eps2 = B * A_hat * dt
    eps1 = beta * (B / B_hat - 1) + Q / R * B_hat * B * dt - A_hat**2 * B / B_hat * dt
    eps0 = Q * B / R * A_hat * dt
    x = sp.symbols('x')

    solutions = sp.solve(- (B + eps2) * x**2 + (-2 * A + beta + eps1) * x + (Q * B / R + eps0), x)
    negatives = [s.evalf() for s in solutions if s.evalf() < 0]
    if negatives:
        K = float(negatives[0])
        P = 1 / (B_hat + B_hat**2 * K * dt + A_hat * B_hat * dt) * R * K
        CA = 1 / (2 * dt) / A * (np.exp(2 * A * dt) - 1)
        gamma = 1 / (beta * dt + 1)
        beta_hat = 1 / dt / gamma - 1 / dt
        return float(K), torch.tensor([P * CA * sigma**2 / beta_hat, 0., P])
    else:
        print("No negative root found.")

def RL_policy_eval_1D(A, B, Q, R, K, sig, beta, dt):
    A_hat = (np.exp(A * dt) - 1) / dt
    B_hat = 1 / A * A_hat * B
    CA = 1 / (2 * dt) / A * (np.exp(2 * A * dt) - 1)
    gamma = 1 / (beta * dt + 1)
    beta_hat = 1 / dt / gamma - 1 / dt

    P = - (Q + K**2 * R) / (beta_hat - (A_hat + B_hat * K) * 2 - (A_hat + B_hat * K)**2 * dt)
    b = sig**2 * P * CA / beta_hat
    return torch.tensor([b, 0., P])


def true_V_eval_1D(A, B, sigma, Q, R, beta, b):
    # returns coefficients of V under basis (1, s, s^2) when applying policy "bs"
    c = 0
    numerator_P = -(Q + R * b**2)
    denominator_P = beta - 2 * (A + B * b)
    P = numerator_P / denominator_P

    # Compute D
    numerator_D = -2 * R * b * c + 2 * P * B * c
    denominator_D = beta - (A + B * b)
    D = numerator_D / denominator_D

    # Compute C
    if beta == 0:
        C = 0
    else:
        C = (-R * c**2 + D * B * c + sigma**2 * P) / beta
    return torch.tensor([C, D, P])

def true_V_eval_2D(A, B, b, R, Q, beta, sigma):

    # Solves for the coefficients P, p, and p0 in the value function V(s) = s^T P s + 2 p^T s + p0
    # Returns: tuple: (P, p, p0)

    # Define symbolic variables for P, p, p0
    c = torch.zeros(2, 1)
    P11, P12, P22 = sp.symbols('P11 P12 P22')  # Only need these since P is symmetric
    p1, p2 = sp.symbols('p1 p2')
    p0 = sp.Symbol('p0')

    # Construct symmetric P matrix
    P = sp.Matrix([[P11, P12], [P12, P22]])
    p = sp.Matrix([[p1], [p2]])

    # Convert PyTorch tensors to SymPy matrices
    A_sp = sp.Matrix(A.tolist())
    B_sp = sp.Matrix(B.tolist())
    b_sp = sp.Matrix(b.tolist())
    c_sp = sp.Matrix(c.tolist())
    R_sp = sp.Matrix(R.tolist())
    Q_sp = sp.Matrix(Q.tolist())

    # Define the Riccati equation for P
    riccati_eq = beta * P - (P * (A_sp + B_sp * b_sp) + (A_sp + B_sp * b_sp).T * P - Q_sp - b_sp.T * R_sp * b_sp)

    # Define the equation for p
    p_eq = beta * p - (P * B_sp * c_sp + (A_sp + B_sp * b_sp).T * p - b_sp.T * R_sp * c_sp)

    # Compute scalar terms explicitly
    cRc = (c_sp.T * R_sp * c_sp)[0, 0]  # Ensure scalar extraction
    pB_c = (p.T * B_sp * c_sp).doit()[0, 0]  # Ensure scalar extraction

    # Define the equation for p0, handling sigma = 0 separately
    if sigma == 0:
        p0_eq = beta * p0 - (-cRc + 2 * pB_c)
    else:
        p0_eq = beta * p0 - (-cRc + 2 * pB_c + sigma**2 * P.trace())

    # Solve for P, p, and p0
    if beta != 0:
        equations = list(riccati_eq) + list(p_eq) + [p0_eq]
        variables = [P11, P12, P22, p1, p2, p0]
        solution = sp.solve(equations, variables, dict=True)

        if not solution:
            print("infeasibility detected")
            return None
    else:
        equations = list(riccati_eq) + list(p_eq)
        variables = [P11, P12, P22, p1, p2]
        solution = sp.solve(equations, variables, dict=True)

        if not solution:
            print("infeasibility detected")
            return None

    # Extract solutions
    sol = solution[0]
    P_sol = torch.tensor([[sol[P11], sol[P12]], [sol[P12], sol[P22]]], dtype=torch.float64)
    p_sol = torch.tensor([[sol[p1]], [sol[p2]]], dtype=torch.float64)
    if beta != 0:
        p0_sol = torch.tensor(sol[p0], dtype=torch.float64)
    else:
        p0_sol = 0.

    return torch.tensor([p0_sol, P_sol[1][1], 2 * P_sol[0][1], P_sol[0][0]])

def is_stabilizable(A, B):
    # Check if (A, B) is stabilizable.
    eigvals, eigvecs = torch.linalg.eig(A)  # Compute eigenvalues of A
    n = A.shape[0]

    for i in range(len(eigvals)):
        if eigvals[i].real >= 0:  # Check unstable eigenvalues
            M = torch.cat((A - eigvals[i] * torch.eye(n), B), dim=1)
            if torch.linalg.matrix_rank(M) < n:
                return False
    return True

def is_detectable(A, C):
    # Check if (A, C) is detectable.
    eigvals, eigvecs = torch.linalg.eig(A)  # Compute eigenvalues of A
    n = A.shape[0]

    for i in range(len(eigvals)):
        if eigvals[i].real >= 0:  # Check unstable eigenvalues
            M = torch.cat((A - eigvals[i] * torch.eye(n), C.T), dim=0)
            if torch.linalg.matrix_rank(M) < n:
                return False
    return True

def LQR_2D_true_solution(A, B, sig, Q, R, beta):
    if (not is_detectable(A - 0.5 * beta * torch.eye(2), B)) or (not is_detectable(A - 0.5 * beta * torch.eye(2), -Q)):
        raise ValueError("Problem not well-posed")
    # value function order: k_00, k_01, k_02, k_10, k_11, k_20
    A_np = A.cpu().numpy().astype(np.float64)
    B_np = B.cpu().numpy().astype(np.float64)
    Q_np = Q.cpu().numpy().astype(np.float64)
    R_np = R.cpu().numpy().astype(np.float64)
    I_np = np.eye(2, dtype=np.float64)


    P = solve_continuous_are(A_np - 0.5 * beta * I_np, B_np, Q_np, R_np)
    # P = solve_continuous_are(A.numpy() - 0.5 * beta * np.eye(2), B.numpy(), Q.numpy(), R.numpy())
    b = - torch.inverse(R) @ B.T @ torch.tensor(P)
    c = torch.zeros(2, 1)
    val_coe = true_V_eval_2D(A, B, b, R, Q, beta, sig)

    return b, c, val_coe

# Merton utilities

def true_solution_merton(mu, r, r_b, sig, gamma):
    # return the optimal constant policy
    p1 = (mu - r) / (gamma * sig**2)
    p2 = (mu - r_b) / (gamma * sig**2)
    if p1 <= 1:
        return p1
    else:
        if p2 <= 1:
            return 1
        else:
            reward_1 = - 0.5 * sig**2 * gamma + mu
            reward_2 = 0.5 * (mu - r_b)**2 / (sig**2 * gamma) + r_b
            if reward_1 >= reward_2:
                return 1
            else:
                return p2

def merton_policy_eval(mu, r, r_b, sig, gamma, running_b, beta):
    # return the coefficient of \sqrt{s}
    if running_b <= 1:
        A = 1 / (1 - gamma) / (- r - running_b * (mu - r) + 0.5 * running_b**2 * sig**2 * gamma + (beta / (1 - gamma)))
        return A / (1 - gamma)
    else:
        A = 1 / (1 - gamma) / (- r_b - running_b * (mu - r_b) + 0.5 * running_b**2 * sig**2 * gamma + (beta / (1 - gamma)))
        return A / (1 - gamma)
def dist_compute_merton(coe, upper=1):
    # l2 distance with in (0, upper)
    return abs(coe) * 0.5 * upper**2


