'''
Exercise 1.4.1 and 1.4.2: Monte Carlo with FOM and ROM

Set p = 9, ell = 5, goal:
    1. Estimate E[Q] by Monte Carlo using the FOM and the ROM, (average across miu ~ U[0.1,10])
    2. and study how the sampling error decreases with M.

The error of FOM came from:
    1. FOM solution is in the N-dim function space, but the temperature func is in the infinite-fim space (first reduction)
    2. algebariac error from using CG to find solution (tolerance)(well-posed: small error in u == small error in Q)
        
 QoI:
     Q(mu)   = q^T * uN(mu)   (FOM)
     Qr(mu)  = qr^T * ur(mu)  (ROM, r = 8)   (Wr^T*A*Wr) * ur = Fr: q = F -> qr = Fr

FOM Monte Carlo (1.4.1):
     Q_ref        = (1 / M_ref) * sum[ Q(mu_i) ]
     s^2          = (1 / (M_ref - 1)) * sum[(Q(mu_i) - Q_ref)^2]   sampling error 
     s_hat        = s / sqrt(M_ref)
     RMSE(M)      ~ sqrt(Var(Q) / M)

ROM Monte Carlo (1.4.2):
     Qr_hat(M)    = (1 / M) * sum_i Q_r(mu_i)
     b_r          = E[Q_r - Q]   which is the lower bound of ROM error
     MSE          = Var(Q_r) / M + b_r^2
     
Need to optimize the running time ??!! Sampling M miu, and apply CG for M times takes looooong time!!!
'''

import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from Q1 import build_model, draw_params, block_layout, qoi,snapshots
from Q3_1 import build_pod_basis, build_reduced_operators, qoi_rom


# ------------------------------------------------------------------------
# Util: QoI mean and standard error
# ------------------------------------------------------------------------
def reference_estimate(model, mus_ref: Array):
    """
    Math:
        Q_ref = (1 / M_ref) * sum_i Q(mu_i)
        s^2   = (1 / (M_ref - 1)) * sum_i (Q(mu_i) - Q_ref)^2
        s_hat = s / sqrt(M_ref)
    """
    # Qs = jnp.array([qoi(model, mu) for mu in mus_ref])   # Qs: (M_ref,)  mus_ref: (M_ref, p), 
    Qs = jax.vmap(lambda mu: qoi(model, mu))(mus_ref)      # Qs: (M_ref,)  使用 vmap 向量化 qoi 函数
    Q_ref = jnp.mean(Qs)
    
    s2 = jnp.var(Qs, ddof=1)          
    s_hat = jnp.sqrt(s2 / mus_ref.shape[0])

    return Q_ref, s2, s_hat


# ------------------------------------------------------------------------
#  主函数: Empirical RMSE for FOM Monte Carlo
# ------------------------------------------------------------------------
def empirical_rmse_fom(model, Q_ref, M_list: list, R: int, key: Array) -> dict:
    """
    for each M:
        repeat R times:
            draw M params, calculate 1. Q_hat_H(M) 2. err = Q_hat_H(M) - Q_ref 3. RMSE(M) = sqrt(mean err^2)

    """
    p = model.block_map.p
    rmse = {}

    for M in M_list:
        errs = []
        for _ in range(R):
            key, subkey = jax.random.split(key)
            mus = draw_params(subkey, M, p)          # (M, p)
            Q_hat = mc_fom(model, mus)               # scalar
            err = Q_hat - Q_ref
            errs.append(err)

        errs = jnp.array(errs)                       # (R,)
        rmse[M] = jnp.sqrt(jnp.mean(errs ** 2))      # RMSE(M)
    return rmse                                      # dict


# Monte Carlo estimators of FOM 
def mc_fom(model, mus: Array) -> Array:
    """
    Q_hat_H(M) = (1 / M) * sum[ Q(mu_i) ]
    """
    # Qs = jnp.array([qoi(model, mu) for mu in mus])   # Qs:(M,) mus:(M,p)
    Qs = jax.vmap(lambda mu: qoi(model, mu))(mus)      # Qs: (M,)  使用 vmap 向量化 qoi 函数
    Q_hat = jnp.mean(Qs)
    return Q_hat # scalar


# ------------------------------------------------------------------------
#  主函数: Empirical RMSE for ROM Monte Carlo
# ------------------------------------------------------------------------
def empirical_rmse_rom(rom: dict, Q_ref, M_list: list, R: int, key: Array) -> dict:
    """
    for each M:
        repeat R times:
            draw M params, 1. Q_hat_r(M) 2. err = Q_hat_r(M) - Q_ref 3. RMSE(M) = sqrt(mean_r err^2)
    """
    W_r = rom["W_r"]
    r = W_r.shape[1]
    p = rom["A_rq"].shape[0]
    rmse = {}

    for M in M_list:
        errs = []
        for _ in range(R):
            key, subkey = jax.random.split(key)
            mus = draw_params(subkey, M, p)          # (M, p)
            Q_hat_r = mc_rom(rom, mus)               # scalar
            err = Q_hat_r - Q_ref
            errs.append(err)

        errs = jnp.array(errs)                       # (R,)
        rmse[M] = jnp.sqrt(jnp.mean(errs ** 2))      # RMSE(M)
    return rmse


# Q_hat_r(M) = (1 / M) * sum[ Q_r(mu_i) ]
def mc_rom(rom: dict, mus: Array) -> Array:
    # Qs = jnp.array([qoi_rom(rom, mu) for mu in mus])  # Qs: (M,) rom: dict
    Qs = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus)     # Qs: (M,)  使用 vmap 向量化 qoi_rom
    Q_hat_r = jnp.mean(Qs)
    return Q_hat_r


def estimate_bias_rom(rom: dict, model, mus_paired: Array) -> Array:
    """
    b_r ~ mean(Qr(mu_i) - Q(mu_i)) = (1 / M) * sum[(Qr(mu_i) - Q(mu_i))]
    E(b_r) = 0 iff ROM ia unbaised estimate
    """
    # Q_r = jnp.array([qoi_rom(rom, mu) for mu in mus_paired])
    # Q_H = jnp.array([qoi(model, mu) for mu in mus_paired])
    
    Q_r = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus_paired)  # 成对样本同时使用 vmap 加速
    Q_H = jax.vmap(lambda mu: qoi(model, mu))(mus_paired)

    b_r = jnp.mean(Q_r - Q_H)
    return b_r


# ------------------------------------------------------------------------
#  Plotting 
# ------------------------------------------------------------------------
# log-log plot of RMSE vs M, plus the curve sqrt(Var(Q) / M)
def plot_rmse_fom(rmse_fom: dict, Q_ref, s2: float, fname: str) -> None:
    """
    Math:
        RMSE(M) ~ sqrt(Var(Q) / M)
    Input:
        rmse_fom : dict M -> RMSE
        Q_ref    : scalar (not used for plotting, kept for interface)
        s2       : scalar, estimate of Var(Q)
        fname    : output file
    """
    M_list = sorted(rmse_fom.keys())
    rmse_vals = jnp.array([rmse_fom[M] for M in M_list])

    # Theoretical curve: sqrt(Var(Q) / M)
    M_arr = jnp.array(M_list, dtype=jnp.float64)
    theory = jnp.sqrt(s2 / M_arr)

    plt.figure(figsize=(7, 5))
    plt.loglog(M_arr, rmse_vals, marker="o", label="empirical RMSE")
    plt.loglog(M_arr, theory, ls="--", label=r"$\sqrt{\mathrm{Var}(Q)/M}$")
    plt.xlabel("M")
    plt.ylabel("RMSE")
    plt.title("FOM Monte Carlo: RMSE vs M")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# mark the bias floor |b_r|
def plot_rmse_fom_vs_rom(rmse_fom: dict,
                         rmse_rom: dict,
                         b_r: Array,
                         s2_rom: float,
                         fname: str) -> None:
    """
    Math:
        FOM:  RMSE(M) ~ sqrt(Var(Q) / M)
        ROM:  RMSE(M) ~ sqrt(Var(Q_r) / M + b_r^2)
    """
    M_list = sorted(rmse_fom.keys())
    M_arr = jnp.array(M_list, dtype=jnp.float64)

    rmse_fom_vals = jnp.array([rmse_fom[M] for M in M_list])
    rmse_rom_vals = jnp.array([rmse_rom[M] for M in M_list])

    # Theoretical ROM curve: sqrt(Var(Q_r) / M + b_r^2)
    theory_rom = jnp.sqrt(s2_rom / M_arr + b_r ** 2)

    plt.figure(figsize=(7, 5))
    plt.loglog(M_arr, rmse_fom_vals, marker="o", label="FOM empirical")
    plt.loglog(M_arr, rmse_rom_vals, marker="s", label="ROM empirical")
    plt.loglog(M_arr, theory_rom, ls="--", label=r"$\sqrt{\mathrm{Var}(Q_r)/M + b_r^2}$")

    # Bias floor
    plt.axhline(abs(float(b_r)), ls=":", color="k",
                label=r"$|b_r|$ (bias floor)")

    plt.xlabel("M")
    plt.ylabel("RMSE")
    plt.title("FOM vs ROM Monte Carlo: RMSE vs M")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# ------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------
def main_exercise_4_1_and_4_2():
    """
    Steps:
        1. Build model for p = 9, ell = 5.
        2. Draw M_ref = 1e4 reference samples.
        3. Compute Q_ref, s^2, s_hat.
        4. Build POD basis with r = 8 from training snapshots.
        5. Build reduced operators.
        6. Estimate b_r from a large paired sample.
        7. For M_list = [10,30,100,300,1000,3000]: empirical RMSE for FOM and ROM
        8. Plot RMSE vs M for FOM and ROM.
    """
    ell = 5
    p = 9
    Bx, By = block_layout(p)
    model = build_model(ell, Bx, By)
    key = jax.random.key(0)

    # Step 2-3: reference estimate
    M_ref = 10_000
    key, subkey = jax.random.split(key)
    mus_ref = draw_params(subkey, M_ref, p)

    Q_ref, s2, s_hat = reference_estimate(model, mus_ref)
    print(f"Q_ref = {float(Q_ref):.6e}")
    print(f"s^2   = {float(s2):.6e}")
    print(f"s_hat = {float(s_hat):.6e}")

    
    # Step 4-5: POD basis and reduced operators
    m_train = 1000
    r = 8
    key, subkey = jax.random.split(key)
    mus_train = draw_params(subkey, m_train, p)
    U_train = snapshots(model, mus_train)

    W_r = build_pod_basis(U_train, r)
    rom = build_reduced_operators(model, W_r)

    # Step 6: estimate ROM bias from paired samples
    M_pair = 2000
    key, subkey = jax.random.split(key)
    mus_paired = draw_params(subkey, M_pair, p)
    b_r = estimate_bias_rom(rom, model, mus_paired)
    print(f"b_r   = {float(b_r):.6e}")

    # Step 7: empirical RMSE for FOM and ROM
    M_list = [10, 30, 100, 300, 1000, 3000]
    R = 20

    key, subkey = jax.random.split(key)
    rmse_fom = empirical_rmse_fom(model, Q_ref, M_list, R, subkey)

    key, subkey = jax.random.split(key)
    rmse_rom = empirical_rmse_rom(rom, Q_ref, M_list, R, subkey)

    print("FOM RMSE:", {M: float(v) for M, v in rmse_fom.items()})
    print("ROM RMSE:", {M: float(v) for M, v in rmse_rom.items()})

    # Step 8: plot RMSE vs M
    plot_rmse_fom(rmse_fom, Q_ref, s2, fname="rmse_fom.pdf")
 
    Q_r_paired = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus_paired) # Var(Q_r) from the paired sample
    s2_rom = jnp.var(Q_r_paired, ddof=1)

    plot_rmse_fom_vs_rom(
        rmse_fom,
        rmse_rom,
        b_r,
        s2_rom,
        fname="rmse_fom_vs_rom.pdf",
    )

    return {
        "Q_ref": Q_ref,
        "s2": s2,
        "s_hat": s_hat,
        "b_r": b_r,
        "rmse_fom": rmse_fom,
        "rmse_rom": rmse_rom,
    }


if __name__ == "__main__":
    main_exercise_4_1_and_4_2()