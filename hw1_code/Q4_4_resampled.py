'''
Exercise 1.4.4 (RESAMPLED version): Multi-fidelity experiment

Same goal as Q4_4.py, but uses pre-built sample pools to avoid solving FOM/ROM
inside the R replications and B budget loop.

Key difference:
    - Build ONE pool of M_pool (Q, Q_r) pairs.
    - mc_fom_budget / mc_rom_budget / mc_cv_budget RESAMPLE from the pool.
    - No more solving inside the RMSE loop.

Math (unchanged):
    Qcv_hat = mean_H^{m_H} + alpha * ( mean_L^{m_L} - mean_L^{m_H} )
    alpha*  = rho * sigma_H / sigma_L
    m_L / m_H = sqrt( rho^2 c_H / ((1 - rho^2) c_L) ),  m_L >= m_H
    Var(Q_cv_hat) = sigma_H^2 * ( (1 - rho^2) / m_H + rho^2 / m_L )
'''

import jax
import jax.numpy as jnp
from jax import Array
import time
import matplotlib.pyplot as plt
from Q1 import build_model, draw_params, snapshots, block_layout, qoi
from Q3_1 import build_pod_basis, build_reduced_operators, qoi_rom


# ------------------------------------------------------------------------
# Build pools
# ------------------------------------------------------------------------
def build_pools(model, rom: dict, M_pool: int, key: Array):
    """
    Build pools of (Q, Q_r) pairs.

    Math:
        Q_pool   = [Q(mu_1),   ..., Q(mu_M)]
        Q_r_pool = [Q_r(mu_1), ..., Q_r(mu_M)]
    """
    p = model.block_map.p
    key, subkey = jax.random.split(key)
    mus = draw_params(subkey, M_pool, p)

    Q_pool = jax.vmap(lambda mu: qoi(model, mu))(mus)
    Q_r_pool = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus)
    return Q_pool, Q_r_pool


# ------------------------------------------------------------------------
# Pilot estimates from the pool (c_H, c_L measured once)
# ------------------------------------------------------------------------
def pilot_from_pool(model, rom, Q_pool, Q_r_pool, m_pilot, key):
    """
    Estimate sigma_H, sigma_L, rho from the pool,
    and c_H, c_L by a short timing run.
    """
    M_pool = Q_pool.shape[0]

    # --- sub-sample from pool for statistics ---
    idx = jax.random.permutation(key, M_pool)[:m_pilot]
    Q_H = Q_pool[idx]
    Q_L = Q_r_pool[idx]

    sigma_H = jnp.std(Q_H)
    sigma_L = jnp.std(Q_L)
    cov = jnp.cov(Q_H, Q_L)[0, 1]
    rho = cov / (sigma_H * sigma_L)

        # --- measure per-sample costs by single-sample timing ---
    p = model.block_map.p
    key, subkey = jax.random.split(key)
    mus = draw_params(subkey, m_pilot, p)

    # FOM: single-sample timing
    t0 = time.perf_counter()
    for i in range(m_pilot):
        Q_temp = qoi(model, mus[i])
        jax.block_until_ready(Q_temp)
    t1 = time.perf_counter()
    c_H = (t1 - t0) / m_pilot

    # ROM: single-sample timing
    t0 = time.perf_counter()
    for i in range(m_pilot):
        Q_temp = qoi_rom(rom, mus[i])
        jax.block_until_ready(Q_temp)
    t1 = time.perf_counter()
    c_L = (t1 - t0) / m_pilot
    
    return {
        "sigma_H": float(sigma_H),
        "sigma_L": float(sigma_L),
        "rho": float(rho),
        "c_H": float(c_H),
        "c_L": float(c_L),
    }


# ------------------------------------------------------------------------
# Optimal alpha and ratio (unchanged)
# ------------------------------------------------------------------------
def optimal_alpha(sigma_H, sigma_L, rho):
    return rho * sigma_H / sigma_L


def optimal_ratio(rho, c_H, c_L):
    ratio = jnp.sqrt((rho ** 2 * c_H) / ((1.0 - rho ** 2) * c_L))
    return float(jnp.maximum(ratio, 1.0))


def allocate_samples(B, c_H, c_L, ratio):
    m_H = B / (c_H + ratio * c_L)
    m_L = ratio * m_H
    return int(max(1, m_H)), int(max(1, m_L))


# ------------------------------------------------------------------------
# Estimators via RESAMPLING from pools
# ------------------------------------------------------------------------
def mc_fom_budget_pooled(Q_pool, B, c_H, key):
    """
    Math:
        M_H = B / c_H
        Q_hat_H = mean(Q_pool[idx]),  idx = random subset of size M_H
    """
    M_pool = Q_pool.shape[0]
    M_H = int(max(1, B / c_H))
    M_H = min(M_H, M_pool)
    idx = jax.random.permutation(key, M_pool)[:M_H]
    return jnp.mean(Q_pool[idx])


def mc_rom_budget_pooled(Q_r_pool, B, c_L, key):
    """
    Math:
        M_L = B / c_L
        Q_hat_r = mean(Q_r_pool[idx])
    """
    M_pool = Q_r_pool.shape[0]
    M_L = int(max(1, B / c_L))
    M_L = min(M_L, M_pool)
    idx = jax.random.permutation(key, M_pool)[:M_L]
    return jnp.mean(Q_r_pool[idx])


def mc_cv_budget_pooled(Q_pool, Q_r_pool, B, c_H, c_L,
                        alpha_star, ratio, key):
    """
    Math:
        Q_cv_hat = mean_H^{m_H}
                 + alpha* ( mean_L^{m_L} - mean_L^{m_H} )
    """
    M_pool = Q_pool.shape[0]
    m_H, m_L = allocate_samples(B, c_H, c_L, ratio)
    m_H = min(m_H, M_pool)
    m_L = min(m_L, M_pool)

    # Draw a permutation and slice it
    perm = jax.random.permutation(key, M_pool)
    idx_H = perm[:m_H]                             # paired samples
    idx_extra = perm[m_H:m_H + max(0, m_L - m_H)]  # extra ROM samples

    Q_H = Q_pool[idx_H]
    Q_L_H = Q_r_pool[idx_H]

    if idx_extra.shape[0] > 0:
        Q_L_extra = Q_r_pool[idx_extra]
    else:
        Q_L_extra = jnp.zeros(0)

    mean_H = jnp.mean(Q_H)
    mean_L_mH = jnp.mean(Q_L_H)
    mean_L_mL = (jnp.sum(Q_L_H) + jnp.sum(Q_L_extra)) / m_L

    Q_cv = mean_H + alpha_star * (mean_L_mL - mean_L_mH)
    return Q_cv


# ------------------------------------------------------------------------
# RMSE helper (unchanged)
# ------------------------------------------------------------------------
def rmse_under_budget(estimator_fn, B, R, Q_ref, key):
    errs = []
    for _ in range(R):
        key, subkey = jax.random.split(key)
        Q_hat = estimator_fn(subkey)
        errs.append(Q_hat - Q_ref)
    errs = jnp.array(errs)
    return float(jnp.sqrt(jnp.mean(errs ** 2)))


# ------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------
def run_exercise_4_4_resampled(ell=5, p=9, r=8,
                               M_pool=10_000,
                               m_pilot=500,
                               R=20,
                               B_list=None,
                               fname="multi_fidelity_rmse_resampled.pdf"):
    if B_list is None:
        B_list = [1.0, 2.0, 4.0, 8.0]

    key = jax.random.key(0)

    # --- model and ROM ---
    Bx, By = block_layout(p)
    model = build_model(ell, Bx, By)

    key, subkey = jax.random.split(key)
    mus_train = draw_params(subkey, 1000, p)
    U_train = snapshots(model, mus_train)
    W_r = build_pod_basis(U_train, r)
    rom = build_reduced_operators(model, W_r)

    # --- build pools (one-shot) ---
    key, subkey = jax.random.split(key)
    Q_pool, Q_r_pool = build_pools(model, rom, M_pool, subkey)

    # --- reference ---
    Q_ref = jnp.mean(Q_pool)

    # --- pilot ---
    key, subkey = jax.random.split(key)
    pilot = pilot_from_pool(model, rom, Q_pool, Q_r_pool, m_pilot, subkey)
    sigma_H, sigma_L, rho = pilot["sigma_H"], pilot["sigma_L"], pilot["rho"]
    c_H, c_L = pilot["c_H"], pilot["c_L"]

    alpha_star = optimal_alpha(sigma_H, sigma_L, rho)
    ratio = optimal_ratio(rho, c_H, c_L)

    # --- bias floor ---
    b_r = jnp.mean(Q_r_pool - Q_pool)

    # --- RMSE under budgets ---
    rmse_fom, rmse_rom, rmse_cv, runtimes = [], [], [], []
    for B in B_list:
        key, subkey = jax.random.split(key)
        rmse_fom.append(rmse_under_budget(
            lambda k: mc_fom_budget_pooled(Q_pool, B, c_H, k),
            B, R, Q_ref, subkey))

        key, subkey = jax.random.split(key)
        rmse_rom.append(rmse_under_budget(
            lambda k: mc_rom_budget_pooled(Q_r_pool, B, c_L, k),
            B, R, Q_ref, subkey))

        key, subkey = jax.random.split(key)
        rmse_cv.append(rmse_under_budget(
            lambda k: mc_cv_budget_pooled(Q_pool, Q_r_pool, B, c_H, c_L,
                                          alpha_star, ratio, k),
            B, R, Q_ref, subkey))

        runtimes.append(B)

    # --- plot ---
    plt.figure(figsize=(7, 5))
    plt.loglog(runtimes, rmse_fom, marker="o", label="FOM-only")
    plt.loglog(runtimes, rmse_rom, marker="s", label="ROM-only")
    plt.loglog(runtimes, rmse_cv, marker="^", label="Control variate")
    plt.axhline(abs(float(b_r)), ls=":", color="k",
                label=r"$|b_r|$ (ROM bias floor)")
    plt.xlabel("total runtime budget")
    plt.ylabel("RMSE")
    plt.title("Multi-fidelity RMSE vs runtime (resampled)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    return {
        "pilot": pilot,
        "alpha_star": alpha_star,
        "ratio": ratio,
        "b_r": float(b_r),
        "rmse_fom": rmse_fom,
        "rmse_rom": rmse_rom,
        "rmse_cv": rmse_cv,
        "runtimes": runtimes,
    }


if __name__ == "__main__":
    results = run_exercise_4_4_resampled()
    print("pilot:", results["pilot"])
    print("alpha*:", results["alpha_star"])
    print("ratio:", results["ratio"])
    print("b_r:", results["b_r"])