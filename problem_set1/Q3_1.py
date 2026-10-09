'''
【Overview】
· Build a POD-based Galerkin reduced-order model (ROM) for the thermal block problem, 
· compare its accuracy against the full-order model (FOM) on an independent test set.

Key objects:
    - POD basis W_r in R^N, obtained from the snapshot matrix U(calculated in Q2).
    - Galerkin ROM:
        A = W^T*A*W; A_r = W_r^T*A_r*W_r
        A_{r,q} = W_r^T A_q W_r,
        F_r     = W_r^T F,
        q_r     = W_r^T q.
    - ROM approximation of the FOM solution:
          u_N(mu) ~ W_r u_r_coaffi(mu).
'''

from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.sparse.linalg import cg
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from Q1 import (build_model, draw_params, snapshots, block_layout, apply_Aq, Model, solve, qoi)


# ------------------------------------------------------------
# 1. 主函数: reduced matrix
# ------------------------------------------------------------
def build_pod_basis(U: Array, r: int) -> Array:
    """
    U = W Sigma V^T, W is square matrix
    W_r = [w_1, ..., w_r] in R^{N x r}     where: W_r^T W_r = I_r; r is obtained from energy tolerence in Q2
    """
    W, _, _ = jnp.linalg.svd(U, full_matrices=False)
    r = min(r, W.shape[1]) # in case of r >= rank
    W_r = W[:, :r]
    return W_r


# ------------------------------------------------------------
# 2. 主函数: the Galerkin condition sum(W_r^T * A_q * W_r * u(miu)) = W_r*F(miu) ---> A_q_r * u(miu) = F_r
#           Calculate "WrT*A_r*W_r * vr"; Ar := sum(mu_q * WrT*A_q*W_r)
# ------------------------------------------------------------
def apply_Ar(rom, mu: Array, v_r: Array) -> Array:
    """
    here is matrix multiplication, not matrix free
    Math:
        A_r(mu) = sum(mu_q * A_rq)
        A_r(mu)* v_r = sum(mu_q * A_rq * v_r) 
    """
    A_rq = rom["A_rq"]
    out = jnp.zeros_like(v_r)
    for q in range(len(mu)):
        out += mu[q] * (A_rq[q] @ v_r)
    return out


# Compute A_q*W_r = [A_q* w_r1, A_q*w_r2, ..., A_q*w_r]  we've written the matrix free func "apply_Aq" in Q1
def apply_Aq_Wr(model:Model, q_idx: int, W_r: Array) -> Array:
    """
    Math: A_q*W_r = [ A_q*w_1, ..., A_q*w_r ] in R^{N x r}, wi is the i-th colume in the matrix W_r of (r,r)
    Input: q_idx : block index q in {0, ..., p-1}
    """
    cols = [apply_Aq(model, q_idx, W_r[:, j]) for j in range(W_r.shape[1])]
    return jnp.stack(cols, axis=1)   # (N, r)

# Compute WrT*A_q*W_r, anf Fr,qr...
def build_reduced_operators(model: Model, W_r: Array) -> dict:
    """
    Math:
        A_q_r  = W_r^T*A_q*W_r,  q = 1, ..., p
        F_r    = W_r^T F
        q_r    = W_r^T q
    Input:  model = Model(mesh=mesh, block_map=block_map, F=F, q=q, K_loc=K_loc, M_loc=M_loc,)
            W_r   = build_pod_basis(U: Array, r: int)
    Output: rom : dict with keys = {A_rq:(p, r, r); F_r:(r,); q_r:(r,); W_r:(N, r)}
    Note: A_q W_r is computed column by column using the matrix-free
    """
    A_rq = []
    for q in range(model.block_map.p):
        A_q_Wr = apply_Aq_Wr(model, q, W_r)   # (N, r)
        A_rq.append(W_r.T @ A_q_Wr)           # (r, r)
    A_rq = jnp.stack(A_rq, axis=0)            # (p, r, r)
    F_r = W_r.T @ model.F
    q_r = W_r.T @ model.q
    return {"A_rq": A_rq, "F_r": F_r, "q_r": q_r, "W_r": W_r}


# ------------------------------------------------------------
# 3. 主函数: find Galerkin solution and QoI
# ------------------------------------------------------------
def solve_rom(rom: dict, mu: Array) -> Array:
    """
    Math:   A_r(mu) u_r(mu) = F_r
            A_r(mu) = sum_q mu_q A_{r,q}
    Input:  rom : ROM dict from build_reduced_operators
            mu  : (p,) parameter vector
    Output: u_r : (r,) reduced solution
    """
    A_rq = rom["A_rq"]           # (p, r, r)
    F_r = rom["F_r"]             # (r,)

    # Assemble A_r(mu) = sum_q mu_q A_{r,q}
    A_r = jnp.einsum("q,qij->ij", mu, A_rq)   # (r, r)
    u_r = jnp.linalg.solve(A_r, F_r)
    return u_r


def qoi_rom(rom: dict, mu: Array) -> Array:
    """
    Math:   Q_r(mu) = q_r^T * u_r(mu) = (W_r^T q)^T * u_r(mu)
    Output: Q_r : scalar
    """
    u_r = solve_rom(rom, mu)
    Q_r = rom["q_r"].T @ u_r
    return Q_r


# ------------------------------------------------------------
# 4. Error measures on the test set
# ------------------------------------------------------------
# The best Euclidean projection error for each test snapshot.
#             (which means it's only a benchemark, can't be obtained)
def projection_error(W_r: Array, U_test: Array) -> Array:
    """
    Math: e_i = |u_i - W_r*W_r^T*u_i| / |u_i|
    Input:  W_r    : (N, r) POD basis
            U_test : (N, m_test) test snapshots
    """
    proj = W_r @ (W_r.T @ U_test)          # (N, m_test)

    err = U_test - proj                    # (N, m_test)
    num = jnp.linalg.norm(err, axis=0)     # (m_test,)
    den = jnp.linalg.norm(U_test, axis=0)  # (m_test,)

    errors = num / den
    return errors                          # (m_test,)

# The relative Galerkin state error for each test parameter.
def galerkin_state_error(rom: dict, mus_test: Array, U_test) -> Array:
    """
    Math: e_i = |uN(mu_i) - Wr*ur(mu_i)| / |uN(mu_i)|
    Input:  model     : Model object
            rom       : ROM dict
            mus_test  : (m_test, p) test parameters
            U_test    : 在外面就算好解, 而不要在这里每次都求解 FOM
    """
    W_r = rom["W_r"]                       # (N, r)

    errors = []
    for i in range(mus_test.shape[0]):
        mu = mus_test[i]

        # Reduced solution and reconstruction
        u_r = solve_rom(rom, mu)           # (r,)
        u_rom = W_r @ u_r                  # (N,)

        # Relative state error
        num = jnp.linalg.norm(U_test[:, i] - u_rom)
        den = jnp.linalg.norm(U_test[:, i])
        errors.append(num / den)
    return jnp.array(errors)               # (m_test,)


# the normalized QoI RMSE over the test set.
def qoi_rmse(rom, mus_test, Q_full):
    """
    Math:   Q(mu)     = q^T * u_N(mu)
            Q_r(mu)   = qr^T * ur(mu);  where qr^T = Wr^T*q, ur(miu) = the Galerkin solution
            e_i       = Q(mu_i) - Q_r(mu_i)
            normalized RMSE  = sqrt( mean_i e_i^2 ) / std(Q)
    """
    Q_rom = jnp.array([qoi_rom(rom, mu) for mu in mus_test])
    e = Q_full - Q_rom
    rmse = jnp.sqrt(jnp.mean(e ** 2))
    
    # return rmse / jnp.std(Q_full)
    return float(rmse / jnp.std(Q_full))


def rms_relative_state_error(rom: dict,
                             mus_test: Array,
                             U_test: Array) -> float:
    """
    Math:   e_i  = |uN(mu_i) - Wr * ur(mu_i)| / |uN(mu_i)|
            RMS  = sqrt( mean_i e_i^2 )
    """
    errors = galerkin_state_error(rom, mus_test, U_test)
    rms_error = jnp.sqrt(jnp.mean(errors ** 2))
    
    #return rms_error
    return float(rms_error)


# ------------------------------------------------------------
# 5. Drivers
# ------------------------------------------------------------
# For every p, build the POD basis and Galerkin ROM, and plot
# the RMS relative state error versus r.
def run_exercise_3_1_state_error(ell: int,
                                 m_train: int,
                                 m_test: int,
                                 p_list: list,
                                 r_list: list,
                                 block_layout_fn: Callable,
                                 fname: str) -> dict:
    """
    Math:   U_train  = snapshots(model, mus_train)
            W_r      = build_pod_basis(U_train, r)
            rom      = build_reduced_operators(model, W_r)
            err(p, r) = rms_relative_state_error(model, rom, mus_test)
    Input:  m_train / m_test: number of training snapshots (>= 1000) / test parameters (>= 250)
            r_list         : list of reduced dimensions, e.g. [1,2,4,6,8,12,16,24,32]
            block_layout_fn: function p -> (Bx, By)
            fname          : output file name for the plot
    Output: dict mapping p -> array of RMS errors over r_list
    """
    results = {}
    key = jax.random.key(0)

    for p in p_list:
        Bx, By = block_layout_fn(p)
        model = build_model(ell, Bx, By)

        # Training snapshots
        key, subkey = jax.random.split(key)
        mus_train = draw_params(subkey, m_train, p)
        U_train = snapshots(model, mus_train)

        # Test parameters
        key, subkey = jax.random.split(key)
        mus_test = draw_params(subkey, m_test, p)
        U_test = snapshots(model, mus_test)

        errors = []
        for r in r_list:
            W_r = build_pod_basis(U_train, r)
            rom = build_reduced_operators(model, W_r)
            err = rms_relative_state_error(rom, mus_test, U_test)
            errors.append(err)
            
        results[p] = jnp.array(errors)
        print(f"p = {p:2d}, errors vs r = {results[p]}")

    plt.figure(figsize=(7, 5))
    for p, errs in results.items():
        plt.semilogy(r_list, errs, marker="o", label=f"p = {p}")
    plt.xlabel("reduced dimension r")
    plt.ylabel("RMS relative state error")
    plt.title(f"Galerkin ROM state error vs r (ell = {ell})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return results


# For p = 9, compare three error curves versus r:
def run_exercise_3_1_p9_comparison(ell: int,
                                   m_train: int,
                                   m_test: int,
                                   p: int,
                                   r_list: list,
                                   block_layout_fn: Callable,
                                   fname: str) -> dict:
    """
    Math: fix p = 9, for each r,
        (a) projection_error(W_r, U_test)
        (b) galerkin_state_error(model, rom, mus_test)
        (c) qoi_rmse(model, rom, mus_test)
    Input:
        ell: mesh level (e.g. 5); m_train>=1000: num of training snapshots; m_test(>= 250)
        p: parameter dimension, i.e. num of thermal blocks;
        r_list: reduced dimensions [1,2,4,6,8,12,..32];
        block_layout_fn: function p -> (Bx, By)
        fname: output file name
    Output:
        results : dict with keys "projection", "state", "qoi"
    """
    Bx, By = block_layout_fn(p)
    model = build_model(ell, Bx, By)
    key = jax.random.key(0)

    # Training snapshots
    key, subkey = jax.random.split(key)
    mus_train = draw_params(subkey, m_train, p)
    U_train = snapshots(model, mus_train)

    # Test snapshots and parameters
    key, subkey = jax.random.split(key)
    mus_test = draw_params(subkey, m_test, p)
    U_test = snapshots(model, mus_test)

    proj_errors = []
    state_errors = []
    qoi_errors = []
    
    Q_full = jnp.array([qoi(model, mu) for mu in mus_test])
    for r in r_list:
        W_r = build_pod_basis(U_train, r)
        rom = build_reduced_operators(model, W_r)

        # (a) Projection error
        e_proj = projection_error(W_r, U_test)
        proj_errors.append(jnp.sqrt(jnp.mean(e_proj ** 2)))

        # (b) Galerkin state error
        e_state = galerkin_state_error(rom, mus_test, U_test)
        state_errors.append(jnp.sqrt(jnp.mean(e_state ** 2)))

        # (c) Normalized QoI RMSE
        e_qoi = qoi_rmse(rom, mus_test, Q_full)
        qoi_errors.append(e_qoi)

    results = {"projection": jnp.array(proj_errors),
            "state": jnp.array(state_errors),
            "qoi": jnp.array(qoi_errors),}

    plt.figure(figsize=(7, 5))
    plt.semilogy(r_list, results["projection"], marker="o", label="projection (benchmark)")
    plt.semilogy(r_list, results["state"], marker="s", label="Galerkin state")
    plt.semilogy(r_list, results["qoi"], marker="^", label="normalized QoI RMSE")
    plt.xlabel("reduced dimension r")
    plt.ylabel("error")
    plt.title(f"p = {p}: projection, state, and QoI errors vs r")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return results


def main_exercise_3_1():
    """
    1. Run run_exercise_3_1_state_error with
        ell=5, m_train>=1000, m_test>=250,
        p_list=[2,4,9,16,25],
        r_list=[1,2,4,6,8,12,16,24,32].
    2. Run run_exercise_3_1_p9_comparison with
        ell=5, p=9, same r_list.
    """
    ell = 5
    m_train = 1000
    m_test = 250
    p_list = [2, 4, 9, 16, 25]
    r_list = [1, 2, 4, 6, 8, 12, 16, 24, 32]

    print("Exercise 1.3.1: Galerkin ROM state error vs r")
    results_state = run_exercise_3_1_state_error(
        ell=ell,
        m_train=m_train,
        m_test=m_test,
        p_list=p_list,
        r_list=r_list,
        block_layout_fn=block_layout,
        fname="rom_state_error_vs_r.pdf",)

    print("Exercise 1.3.1: p = 9 comparison")
    results_p9 = run_exercise_3_1_p9_comparison(
        ell=ell,
        m_train=m_train,
        m_test=m_test,
        p=9,
        r_list=r_list,
        block_layout_fn=block_layout,
        fname="rom_p9_comparison.pdf",)
    return results_state, results_p9


if __name__ == "__main__":
    main_exercise_3_1()