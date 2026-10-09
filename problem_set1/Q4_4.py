''' 
Set p = 9, ell = 5, r = 8; mu_q ~ Unif[0.1, 10], iid. Compare three Monte Carlo estimators and plot RMSE vs runtime.
         (a) FOM-only MC
         (b) ROM-only MC
         (c) FOM-ROM control variate with alpha* and optimal (m_H, m_L)
     
Definitions:
     Q        = FOM QoI,   Var(Q)   = sigma_H^2
     Q_r      = ROM QoI,   Var(Q_r) = sigma_L^2
     rho      = Corr(Q, Q_r)
     c_H, c_L = per-sample costs of FOM and ROM

Math:
Qcv_hat = mean_H^{m_H} + alpha * ( mean_L^{m_L} - mean_L^{m_H} )
    optimal alpha* = Cov(Q, Q_r) / Var(Q_r)
    optimal m_L / m_H = sqrt( rho^2 c_H / ((1 - rho^2) c_L) ),  m_L >= m_H
Var(Q_cv_hat) = sigma_H^2 * ( (1 - rho^2) / m_H + rho^2 / m_L )

Use resampling to accelerate, use M times of FOM\ROM as the pool!!
'''

import jax
import jax.numpy as jnp
from jax import Array
import time
import matplotlib.pyplot as plt
from Q1 import build_model, draw_params, snapshots, block_layout, qoi
from Q3_1 import build_pod_basis, build_reduced_operators, qoi_rom
from Q4_1_2 import reference_estimate, estimate_bias_rom


# ---------------------------------------------------------------------------------
# Util: estimate parameters sigma_H, sigma_L, rho, c_H, c_L, optimal alpha, optimal ratio
# ---------------------------------------------------------------------------------
# Estimate sigma_H, sigma_L, rho, c_H, c_L from pilot data.
def pilot_estimates(model, rom: dict, m_pilot: int, key: Array) -> dict:
    """
    sigma_H^2 = Var(Q)
    sigma_L^2 = Var(Q_r)
    rho       = Corr(Q, Q_r)
    c_H       = time of one FOM solve + QoI
    c_L       = time of one ROM solve + QoI
    """
    p = model.block_map.p
    key, subkey = jax.random.split(key)
    mus = draw_params(subkey, m_pilot, p)
    
    # --- FOM pilot ---
    t0 = time.perf_counter()
    Q_H = jax.vmap(lambda mu: qoi(model, mu))(mus) # (m_pilot,)
    jax.block_until_ready(Q_H)
    t1 = time.perf_counter()
    c_H = (t1 - t0) / m_pilot

    # --- ROM pilot ---
    t0 = time.perf_counter()
    Q_L = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus) # (m_pilot,)
    jax.block_until_ready(Q_L)
    t1 = time.perf_counter()
    c_L = (t1 - t0) / m_pilot

    sigma_H = jnp.std(Q_H)
    sigma_L = jnp.std(Q_L)
    cov = jnp.cov(Q_H, Q_L)[0, 1]
    rho = cov / (sigma_H * sigma_L)

    return {
        "sigma_H": float(sigma_H),
        "sigma_L": float(sigma_L),
        "rho": float(rho),
        "c_H": float(c_H),
        "c_L": float(c_L),}
    

# optimal alpha to minimize Var(Qcv)
def optimal_alpha(sigma_H: float,
                  sigma_L: float,
                  rho: float) -> float:
    """
    alpha* = Cov(Q, Q_r) / Var(Q_r)  = rho * sigma_H / sigma_L
    """
    return rho * sigma_H / sigma_L


# optimal ratio to minimize Var(Qcv)
def optimal_ratio(rho: float,
                  c_H: float,
                  c_L: float) -> float:
    """
    m_L / m_H = sqrt( rho^2 c_H / ((1 - rho^2) c_L) )
    """
    ratio = jnp.sqrt((rho ** 2 * c_H) / ((1.0 - rho ** 2) * c_L))
    return float(jnp.maximum(ratio, 1.0))

def allocate_samples(B: float,
                     c_H: float,
                     c_L: float,
                     ratio: float) -> tuple:
    """
    m_H * c_H + m_L * c_L = B
    by Lagrange:   m_H = B / (c_H + ratio * c_L)
                m_L = ratio * m_H
    """
    m_H = B / (c_H + ratio * c_L)
    m_L = ratio * m_H
    return int(max(1, m_H)), int(max(1, m_L))


# ---------------------------------------------------------------------------------
# Estimators Q for FOM, ROM and the control version of ROM, under time budget B 
# ---------------------------------------------------------------------------------
def mc_fom_budget(model, B: float, c_H: float, key: Array) -> Array:
    """
    Math:
        M_H = B / c_H
        Q_hat_H = (1 / M_H) * sum_i Q(mu_i)
    Input:
        model : Model
        B     : budget
        c_H   : FOM cost
        key   : JAX random key
    Output:
        Q_hat : scalar
    """
    M_H = int(max(1, B / c_H))
    p = model.block_map.p
    mus = draw_params(key, M_H, p)
    Qs = jax.vmap(lambda mu: qoi(model, mu))(mus)
    return jnp.mean(Qs)


def mc_rom_budget(rom: dict, B: float, c_L: float, p: int, key: Array) -> Array:
    """
    Math:
        M_L = B / c_L
        Q_hat_r = (1 / M_L) * sum_i Q_r(mu_i)
    Input:
        rom : ROM dict
        B   : budget
        c_L : ROM cost
        p   : parameter dimension
        key : JAX random key
    Output:
        Q_hat_r : scalar
    """
    M_L = int(max(1, B / c_L))
    mus = draw_params(key, M_L, p)
    Qs = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus)
    return jnp.mean(Qs)


# calculate avg Qcv, where Qcv_hat = Qmean_H^mH + alpha* ( Qmean_L^mL} - Qmean_L^mH} )
def mc_cv_budget(model, rom: dict,
                 B: float, c_H: float, c_L: float,
                 alpha_star: float,ratio: float,
                 key: Array) -> Array:
    
    p = model.block_map.p
    m_H, m_L = allocate_samples(B, c_H, c_L, ratio)

    # Paired samples: same mu for FOM and ROM
    key, subkey = jax.random.split(key)
    mus_H = draw_params(subkey, m_H, p)

    Q_H = jax.vmap(lambda mu: qoi(model, mu))(mus_H)      # (m_H,)
    Q_L_H = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus_H)  # (m_H,)
    ...

    # Extra ROM-only samples
    m_extra = max(0, m_L - m_H)
    if m_extra > 0:
        key, subkey = jax.random.split(key)
        mus_extra = draw_params(subkey, m_extra, p)
        Q_L_extra = jax.vmap(lambda mu: qoi_rom(rom, mu))(mus_extra)
    else:
        Q_L_extra = jnp.zeros(0)

    mean_H = jnp.mean(Q_H)
    mean_L_mH = jnp.mean(Q_L_H)
    mean_L_mL = (jnp.sum(Q_L_H) + jnp.sum(Q_L_extra)) / m_L

    Q_cv = mean_H + alpha_star * (mean_L_mL - mean_L_mH)
    return Q_cv

# ---------------------------------------------------------------------------------
# Util: RMSE under time budget B 通用函数 不针对特定 Qcv/QL/QH 
# ---------------------------------------------------------------------------------
def rmse_under_budget(estimator_fn,
                      B: float,
                      R: int,
                      Q_ref: float,
                      key: Array) -> float:
    """
    RMSE(B) = sqrt( (1 / R) * sum_r (Q_hat^{(r)} - Q_ref)^2 )
    """
    errs = []
    for _ in range(R):
        key, subkey = jax.random.split(key)
        Q_hat = estimator_fn(subkey)
        errs.append(Q_hat - Q_ref)
    errs = jnp.array(errs)
    return float(jnp.sqrt(jnp.mean(errs ** 2)))


# ---------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------
def run_exercise_4_4(ell: int = 5,
                     p: int = 9,
                     r: int = 8,
                     m_pilot: int = 500,
                     m_ref: int = 10_000,
                     R: int = 20,
                     B_list: list = None,
                     fname: str = "multi_fidelity_rmse.pdf"):
    """
    1. Build model and ROM (p = 9, ell = 5, r = 8).
    2. Compute reference Q_ref from m_ref FOM samples.
    3. Pilot estimates: sigma_H, sigma_L, rho, c_H, c_L.
    4. alpha_star, ratio.
    5. For each time budget B: - RMSE for FOM-only, ROM-only, control variate
    6. Plot RMSE vs total runtime.
    """
    if B_list is None:
        B_list = [1.0, 2.0, 4.0, 8.0]

    key = jax.random.key(0)

    # Step 1: model and ROM
    Bx, By = block_layout(p)
    model = build_model(ell, Bx, By)

    key, subkey = jax.random.split(key)
    mus_train = draw_params(subkey, 1000, p)
    U_train = snapshots(model, mus_train)
    W_r = build_pod_basis(U_train, r)
    rom = build_reduced_operators(model, W_r)

    # Step 2: reference
    key, subkey = jax.random.split(key)
    mus_ref = draw_params(subkey, m_ref, p)
    Q_ref, _, _ = reference_estimate(model, mus_ref)

    # Step 3: pilot estimates
    key, subkey = jax.random.split(key)
    pilot = pilot_estimates(model, rom, m_pilot, subkey)

    sigma_H = pilot["sigma_H"]
    sigma_L = pilot["sigma_L"]
    rho = pilot["rho"]
    c_H = pilot["c_H"]
    c_L = pilot["c_L"]

    # Step 4: alpha* and ratio
    alpha_star = optimal_alpha(sigma_H, sigma_L, rho)
    ratio = optimal_ratio(rho, c_H, c_L)

    # Bias magnitude of ROM-only
    key, subkey = jax.random.split(key)
    mus_pair = draw_params(subkey, 2000, p)
    b_r = estimate_bias_rom(rom, model, mus_pair)

    # Step 5: RMSE under budgets
    rmse_fom = []
    rmse_rom = []
    rmse_cv = []
    runtimes = []

    for B in B_list:
        key, subkey = jax.random.split(key)

        rmse_fom.append(rmse_under_budget(
            lambda k: mc_fom_budget(model, B, c_H, k),
            B, R, Q_ref, subkey))

        key, subkey = jax.random.split(key)
        rmse_rom.append(rmse_under_budget(
            lambda k: mc_rom_budget(rom, B, c_L, p, k), B, R, Q_ref, subkey))

        key, subkey = jax.random.split(key)
        rmse_cv.append(rmse_under_budget(
            lambda k: mc_cv_budget(model, rom, B, c_H, c_L,
                                   alpha_star, ratio, k),
                                   B, R, Q_ref, subkey))
        runtimes.append(B)

    # Step 6: plot
    plt.figure(figsize=(7, 5))
    plt.loglog(runtimes, rmse_fom, marker="o", label="FOM-only")
    plt.loglog(runtimes, rmse_rom, marker="s", label="ROM-only")
    plt.loglog(runtimes, rmse_cv, marker="^", label="Control variate")
    plt.axhline(abs(float(b_r)), ls=":", color="k",
                label=r"$|b_r|$ (ROM bias floor)")
    plt.xlabel("total runtime budget")
    plt.ylabel("RMSE")
    plt.title("Multi-fidelity RMSE vs runtime")
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
        "runtimes": runtimes,}


if __name__ == "__main__":
    results = run_exercise_4_4()
    print("pilot:", results["pilot"])
    print("alpha*:", results["alpha_star"])
    print("ratio:", results["ratio"])
    print("b_r:", results["b_r"])