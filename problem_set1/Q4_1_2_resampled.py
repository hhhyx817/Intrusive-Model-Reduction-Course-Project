'''
Exercise 1.4.1 and 1.4.2 (RESAMPLED version): Monte Carlo with FOM and ROM

Same goal as Q4_1_2.py, but uses a pre-built sample pool to avoid solving FOM repeatedly.

Key difference:
    - Build ONE pool of M_pool (Q, Q_r) pairs (one FOM + one ROM evaluation each).
    - All later empirical RMSE experiments RESAMPLE from the pool.
    - No more solving FOM inside the M/R loops.

Math:
    Q_ref        = (1 / M_pool) * sum_i Q(mu_i)
    s^2          = (1 / (M_pool - 1)) * sum[(Q(mu_i) - Q_ref)^2]
    s_hat        = s / sqrt(M_pool)
    RMSE(M)      ~ sqrt(Var(Q) / M)
    b_r          = mean(Q_r - Q)
    MSE_ROM      = Var(Q_r) / M + b_r^2
'''

import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from Q1 import build_model, draw_params, block_layout, qoi, snapshots
from Q3_1 import build_pod_basis, build_reduced_operators, qoi_rom


# ------------------------------------------------------------------------
# Build the sample pool
# ------------------------------------------------------------------------
def build_pools(model, rom: dict, M_pool: int, key: Array):
    """
    Build pools of (Q, Q_r) pairs, one evaluation per mu.

    Math:
        Q_pool    = [Q(mu_1), ..., Q(mu_M)]
        Q_r_pool  = [Q_r(mu_1), ..., Q_r(mu_M)]

    Input:
        model  : Model
        rom    : ROM dict
        M_pool : pool size
        key    : JAX random key
    Output:
        Q_pool   : (M_pool,)
        Q_r_pool : (M_pool,)
    """
    p = model.block_map.p
    key, subkey = jax.random.split(key)
    mus = draw_params(subkey, M_pool, p)

    Q_pool = jax.vmap(lambda mu: qoi(model, mu))(mus)
    Q_r_pool = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus)

    return Q_pool, Q_r_pool


# ------------------------------------------------------------------------
# Reference estimate (from pool)
# ------------------------------------------------------------------------
def reference_from_pool(Q_pool: Array):
    """
    Math:
        Q_ref = (1 / M) * sum_i Q(mu_i)
        s^2   = (1 / (M - 1)) * sum_i (Q(mu_i) - Q_ref)^2
        s_hat = s / sqrt(M)
    """
    Q_ref = jnp.mean(Q_pool)
    s2 = jnp.var(Q_pool, ddof=1)
    s_hat = jnp.sqrt(s2 / Q_pool.shape[0])
    return Q_ref, s2, s_hat


# ------------------------------------------------------------------------
# Empirical RMSE by RESAMPLING from pool
# ------------------------------------------------------------------------
def empirical_rmse_fom_pooled(Q_pool: Array,
                              Q_ref: float,
                              M_list: list,
                              R: int,
                              key: Array) -> dict:
    """
    Math:
        for each M:
            repeat R times:
                idx   = random subset of size M (no replacement)
                Q_hat = mean(Q_pool[idx])
                err   = Q_hat - Q_ref
            RMSE(M) = sqrt(mean_r err^2)
    """
    M_pool = Q_pool.shape[0]
    rmse = {}

    for M in M_list:
        M = min(M, M_pool)
        errs = []
        for _ in range(R):
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, M_pool)[:M]
            Q_hat = jnp.mean(Q_pool[idx])
            errs.append(Q_hat - Q_ref)
        errs = jnp.array(errs)
        rmse[M] = jnp.sqrt(jnp.mean(errs ** 2))
    return rmse


def empirical_rmse_rom_pooled(Q_r_pool: Array,
                              Q_ref: float,
                              M_list: list,
                              R: int,
                              key: Array) -> dict:
    """
    Math:
        for each M:
            repeat R times:
                idx   = random subset of size M
                Q_hat = mean(Q_r_pool[idx])
                err   = Q_hat - Q_ref
            RMSE(M) = sqrt(mean_r err^2)
    """
    M_pool = Q_r_pool.shape[0]
    rmse = {}

    for M in M_list:
        M = min(M, M_pool)
        errs = []
        for _ in range(R):
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, M_pool)[:M]
            Q_hat = jnp.mean(Q_r_pool[idx])
            errs.append(Q_hat - Q_ref)
        errs = jnp.array(errs)
        rmse[M] = jnp.sqrt(jnp.mean(errs ** 2))
    return rmse


# ------------------------------------------------------------------------
# Bias estimation (from pool)
# ------------------------------------------------------------------------
def estimate_bias_rom_pooled(Q_pool: Array, Q_r_pool: Array) -> Array:
    """
    Math:
        b_r ~ mean(Q_r - Q)
    """
    return jnp.mean(Q_r_pool - Q_pool)


# ------------------------------------------------------------------------
# Plotting (unchanged from original)
# ------------------------------------------------------------------------
def plot_rmse_fom(rmse_fom: dict, Q_ref, s2: float, fname: str) -> None:
    M_list = sorted(rmse_fom.keys())
    rmse_vals = jnp.array([rmse_fom[M] for M in M_list])

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


def plot_rmse_fom_vs_rom(rmse_fom: dict,
                         rmse_rom: dict,
                         b_r: Array,
                         s2_rom: float,
                         fname: str) -> None:
    M_list = sorted(rmse_fom.keys())
    M_arr = jnp.array(M_list, dtype=jnp.float64)

    rmse_fom_vals = jnp.array([rmse_fom[M] for M in M_list])
    rmse_rom_vals = jnp.array([rmse_rom[M] for M in M_list])
    theory_rom = jnp.sqrt(s2_rom / M_arr + b_r ** 2)

    plt.figure(figsize=(7, 5))
    plt.loglog(M_arr, rmse_fom_vals, marker="o", label="FOM empirical")
    plt.loglog(M_arr, rmse_rom_vals, marker="s", label="ROM empirical")
    plt.loglog(M_arr, theory_rom, ls="--", label=r"$\sqrt{\mathrm{Var}(Q_r)/M + b_r^2}$")
    plt.axhline(abs(float(b_r)), ls=":", color="k", label=r"$|b_r|$ (bias floor)")

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
def main_exercise_4_1_and_4_2_resampled():
    """
    Steps:
        1. Build model and ROM (p=9, ell=5, r=8).
        2. Build the sample pool (Q_pool, Q_r_pool).
        3. Reference estimate from pool.
        4. Bias from pool.
        5. Empirical RMSE for FOM and ROM by resampling.
        6. Plot.
    """
    ell = 5
    p = 9
    Bx, By = block_layout(p)
    model = build_model(ell, Bx, By)
    key = jax.random.key(0)

    # --- ROM construction ---
    m_train = 1000
    r = 8
    key, subkey = jax.random.split(key)
    mus_train = draw_params(subkey, m_train, p)
    U_train = snapshots(model, mus_train)
    W_r = build_pod_basis(U_train, r)
    rom = build_reduced_operators(model, W_r)

    # --- Build pool (one-shot cost) ---
    M_pool = 10_000
    key, subkey = jax.random.split(key)
    Q_pool, Q_r_pool = build_pools(model, rom, M_pool, subkey)

    # --- Reference ---
    Q_ref, s2, s_hat = reference_from_pool(Q_pool)
    print(f"Q_ref = {float(Q_ref):.6e}")
    print(f"s^2   = {float(s2):.6e}")
    print(f"s_hat = {float(s_hat):.6e}")

    # --- Bias ---
    b_r = estimate_bias_rom_pooled(Q_pool, Q_r_pool)
    print(f"b_r   = {float(b_r):.6e}")

    # --- Empirical RMSE by resampling ---
    M_list = [10, 30, 100, 300, 1000, 3000]
    R = 20

    key, subkey = jax.random.split(key)
    rmse_fom = empirical_rmse_fom_pooled(Q_pool, Q_ref, M_list, R, subkey)

    key, subkey = jax.random.split(key)
    rmse_rom = empirical_rmse_rom_pooled(Q_r_pool, Q_ref, M_list, R, subkey)

    print("FOM RMSE:", {M: float(v) for M, v in rmse_fom.items()})
    print("ROM RMSE:", {M: float(v) for M, v in rmse_rom.items()})

    # --- Plots ---
    plot_rmse_fom(rmse_fom, Q_ref, s2, fname="rmse_fom_resampled.pdf")
    s2_rom = jnp.var(Q_r_pool, ddof=1)
    plot_rmse_fom_vs_rom(rmse_fom, rmse_rom, b_r, s2_rom,
                         fname="rmse_fom_vs_rom_resampled.pdf")

    return {
        "Q_ref": Q_ref, "s2": s2, "s_hat": s_hat, "b_r": b_r,
        "rmse_fom": rmse_fom, "rmse_rom": rmse_rom,
    }


if __name__ == "__main__":
    main_exercise_4_1_and_4_2_resampled()