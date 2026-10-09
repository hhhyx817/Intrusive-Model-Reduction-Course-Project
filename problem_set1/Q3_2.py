'''
Goal:
    Measure the online runtime of the FOM and the ROM, 
    combine runtime with accuracy, and estimate the break-even number of online queries.
    即构造 ROM 之后,评估它的精度、速度, 以及“用多少次才回本”。

ell = 5, carry both p = 9 and p = 25. To measure separately:
    (1) one matrix-free FOM application A(mu)* v (offline)
    (2) full FOM solve + QoI (online)
    (3) ROM online parts:
         - affine reduced-operator combination
         - dense reduced solve
         - reduced QoI evaluation

 Plots:
     - online runtime vs r, for each p. 
       ROM rumtime increases as r increases, and at r=r0, ROM runtime may exceed FOM runtime
     - runtime vs error, combining timing data with test errors

 Break-even:
    N_BE = C_off / (t_H - t_r),  when t_H > t_r 
    C_off = Offline cost = (snapshot generation) + (POD) + (reduced-operator construction)
    即查询多少次, 离线构造 + 查询次数 * ROM查询时间 > 查询次数 * FOM查询时间
'''
from typing import Callable
import time
import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from Q1 import (build_model,draw_params,snapshots,block_layout,apply_A,solve,qoi,)
from Q3_1 import (build_pod_basis,build_reduced_operators,solve_rom,qoi_rom,galerkin_state_error,qoi_rmse,)


# ------------------------------------------------------------
# 1. Timing utilities
# ------------------------------------------------------------
# Synchronize a JAX result to make sure the computation is done.
def sync(x):
    """
    Output: x : same array, after blocking until computation finishes
    """
    return jax.block_until_ready(x)


# Measure the median runtime of a function.
def median_time(fn: Callable, *args, repeats: int = 10, warmup: int = 3, **kwargs):
    """
    Input:  fn      : function to time
            args    : positional arguments
            repeats : number of timing repetitions
            warmup  : number of warm-up runs
            kwargs  : keyword arguments
    """
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        sync(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        sync(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = jnp.array(times)
    t_med = float(jnp.median(times))   # median( t_1, ..., t_R )
    return t_med                       # in seconds


# Break-even 即多少查询的时候, ROM 比 FOM 时间更值得
def break_even(C_off, t_H, t_r):
    """
    Input:  C_off : offline cost (seconds)
            t_H   : FOM online time (seconds)
            t_r   : ROM online time (seconds)
    """
    if t_H <= t_r:
        return jnp.inf
    return C_off / (t_H - t_r) # break-even query count (float or jnp.inf)


# ------------------------------------------------------------
# 2. Time of FOM and ROM components / 计算 FOM 和 ROM 相关步骤的时间
# ------------------------------------------------------------
# time(matrix-free FOM application: v -> A(mu)*v, by local stiffness matrices Kloc_element)
def time_fom_apply(model, mu, v, repeats=10, warmup=3):
    fn = lambda v_: apply_A(model, mu, v_)  # v: (N,) ; mu: (p,)
    return median_time(fn, v, repeats=repeats, warmup=warmup)


# time(full FOM solve: A(mu)*u = F) + time(QoI evaluation: Q = q^T u)    
def time_fom_solve_qoi(model, mu, repeats=10, warmup=3):
    def fn():
        u = solve(model, mu)  # mu: (p,)
        Q = qoi(model, mu)
        return u, Q
    return median_time(fn, repeats=repeats, warmup=warmup)


# time(affine reduced-operator combination: Ar(mu) = sum(mu_q * A_rq),
#                                           Ar(mu)*vr = sum(mu_q * A_rq * vr) )
def time_rom_affine_combination(rom, mu, v_r, repeats=10, warmup=3):
    A_rq = rom["A_rq"]                      # rom : ROM dict

    def fn(v_):                             # v_r : (r,)
        out = jnp.zeros_like(v_)
        for q in range(len(mu)):            # mu  : (p,)
            out = out + mu[q] * (A_rq[q] @ v_)
        return out
    return median_time(fn, v_r, repeats=repeats, warmup=warmup)


# time(find the Galerkin solve: Ar(mu) * ur = Fr)
def time_rom_solve(rom, mu, repeats=10, warmup=3):
    def fn():
        return solve_rom(rom, mu)

    return median_time(fn, repeats=repeats, warmup=warmup)


# time(reduced QoI evaluation: Qr(mu) = qr^T * ur(mu) )
def time_rom_qoi(rom, mu, repeats=10, warmup=3):
    def fn():
        return qoi_rom(rom, mu)

    return median_time(fn, repeats=repeats, warmup=warmup)


"""
time of the offline cost: snapshot generation: U_train  = snapshots(model, mus_train) 
                        + POD: W_r = build_pod_basis(U_train, r)
                        + reduced-operator construction: rom = build_reduced_operators(model, W_r)      
"""
def time_offline_cost(model, m_train, r, key):
    p = model.block_map.p
 
    t0 = time.perf_counter()

    mus_train = draw_params(key, m_train, p) # key:JAX random key, m_train: num of training snapshots
    U_train = snapshots(model, mus_train)
    sync(U_train)

    W_r = build_pod_basis(U_train, r)
    sync(W_r)

    rom = build_reduced_operators(model, W_r)
    sync(rom["A_rq"])

    t1 = time.perf_counter()
    return t1 - t0



# ============================================================
# 3. Driver1: runtime vs r
# ============================================================
def run_exercise_3_2_runtime(ell: int,
                             m_train: int,
                             m_test: int,
                             p_list: list,
                             r_list: list,
                             block_layout_fn: Callable,
                             fname: str) -> dict:
    """
    For each p, measure:
        - FOM apply time
        - FOM solve + QoI time
        - ROM component times vs r
    
    Math:   t_H = time(solve + qoi) 即 FOM 求解的时间, 此外 FOM 还有求 A*v的时间
            t_r(r) = time( affine combination + dense solve + QoI ) 即 ROM 的时间
    """
    results = {}
    key = jax.random.key(0)

    for p in p_list:
        Bx, By = block_layout_fn(p)
        model = build_model(ell, Bx, By)

        mu = 0.5 * jnp.ones(p) + 0.5  # a representative parameter
        v = jnp.ones(model.mesh.N)

        # FOM timings
        t_apply = time_fom_apply(model, mu, v)
        t_fom = time_fom_solve_qoi(model, mu)

        # Training snapshots for ROM
        key, subkey = jax.random.split(key)
        mus_train = draw_params(subkey, m_train, p)
        U_train = snapshots(model, mus_train)

        rom_times = []
        for r in r_list:
            W_r = build_pod_basis(U_train, r)
            rom = build_reduced_operators(model, W_r)

            mu_r = mu
            v_r = jnp.ones(r)

            t_aff = time_rom_affine_combination(rom, mu_r, v_r)
            t_solve = time_rom_solve(rom, mu_r)
            t_qoi = time_rom_qoi(rom, mu_r)

            t_rom = t_aff + t_solve + t_qoi
            rom_times.append(t_rom)
        results[p] = {"t_apply": t_apply, "t_fom": t_fom, "t_rom": jnp.array(rom_times),}

    plt.figure(figsize=(7, 5))
    for p, data in results.items():
        plt.semilogy(r_list, data["t_rom"], marker="o", label=f"ROM p = {p}")
        plt.axhline(data["t_fom"], ls="--", label=f"FOM p = {p}")
    plt.xlabel("reduced dimension r")
    plt.ylabel("online runtime (s)")
    plt.title(f"Online runtime vs r (ell = {ell})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    return results # dict


# ============================================================
# Driver2: runtime vs error, and break-even
# ============================================================
def run_exercise_3_2_runtime_vs_error(ell: int,
                                     m_train: int,
                                     m_test: int,
                                     p_list: list,
                                     r_list: list,
                                     block_layout_fn: Callable,
                                     fname: str) -> dict:
    """
    Combine runtime and accuracy:
        - compute ROM errors vs r
        - compute online times vs r
        - estimate break-even N_BE at the smallest r with normalized QoI RMSE < 1e-3

    Math:   err(r) = qoi_rmse(...)
            t_r(r) = online ROM time
            N_BE   = C_off / (t_H - t_r)
    """
    summary = {}
    key = jax.random.key(0)

    for p in p_list:
        Bx, By = block_layout_fn(p)
        model = build_model(ell, Bx, By)

        # Training snapshots
        key, subkey = jax.random.split(key)
        mus_train = draw_params(subkey, m_train, p)
        U_train = snapshots(model, mus_train)

        # Test snapshots and QoI
        key, subkey = jax.random.split(key)
        mus_test = draw_params(subkey, m_test, p)
        U_test = snapshots(model, mus_test)
        Q_full = jnp.array([qoi(model, mu) for mu in mus_test])

        mu = 0.5 * jnp.ones(p) + 0.5

        # FOM time
        t_fom = time_fom_solve_qoi(model, mu)

        errors = []
        times = []
        
        # ★ DEBUG
        print(f"p = {p}, Q_full shape = {Q_full.shape}, std = {float(jnp.std(Q_full))}, mean = {float(jnp.mean(Q_full))}")
        
        for r in r_list:
            W_r = build_pod_basis(U_train, r)
            rom = build_reduced_operators(model, W_r)

            # Error
            err = qoi_rmse(rom, mus_test, Q_full)
            errors.append(err)
            
             # ★ DEBUG: 打印每个 r 对应的 QoI RMSE
            print(f"  [p={p}, r={r}] err = {err:.4e}")

            # Online time
            v_r = jnp.ones(r)
            t_aff = time_rom_affine_combination(rom, mu, v_r)
            t_solve = time_rom_solve(rom, mu)
            t_qoi = time_rom_qoi(rom, mu)
            times.append(t_aff + t_solve + t_qoi)

        
        errors = jnp.array(errors)
        # ★ DEBUG
        print(f"p = {p}, errors = {errors}")   
        print(f"p = {p}, min(errors) = {float(jnp.min(errors))}, idx = {jnp.where(errors < 1e-3, size=1)[0]}")   
        times = jnp.array(times)

        # Offline cost at the smallest r with err < 1e-3        
        idx = jnp.where(errors < 1e-3)[0]
        if idx.size > 0:
            r_star = r_list[int(idx[0])]
        else:
            r_star = r_list[-1]

        key, subkey = jax.random.split(key)
        C_off = time_offline_cost(model, m_train, r_star, subkey)

        t_r_star = times[r_list.index(r_star)]
        N_BE = break_even(C_off, t_fom, t_r_star)

        summary[p] = {"r_star": r_star, "C_off": C_off,"t_fom": t_fom,
                      "t_r": t_r_star,"N_BE": N_BE,"errors": errors,"times": times,}

    # Plot runtime vs error
    plt.figure(figsize=(7, 5))
    for p, data in summary.items():
        plt.loglog(data["times"], data["errors"], marker="o", label=f"p = {p}")
    plt.xlabel("online runtime (s)")
    plt.ylabel("normalized QoI RMSE")
    plt.title(f"Runtime vs error (ell = {ell})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    return summary # dict with per-p break-even data


# ============================================================
# Main driver
# ============================================================
def main_exercise_3_2():
    """
    1. Run runtime vs r for p in {9, 25}.
    2. Run runtime vs error and report break-even N_BE.
    """
    ell = 5
    m_train = 1000
    m_test = 250
    p_list = [9, 25]
    r_list = [1, 2, 4, 6, 8, 12, 16, 24, 32]

    print("Exercise 1.3.2: runtime vs r")
    results_rt = run_exercise_3_2_runtime(
        ell=ell,
        m_train=m_train, m_test=m_test,
        p_list=p_list, r_list=r_list,
        block_layout_fn=block_layout, fname="rom_runtime_vs_r.pdf",)

    print("Exercise 1.3.2: runtime vs error and break-even")
    summary = run_exercise_3_2_runtime_vs_error(
        ell=ell,
        m_train=m_train, m_test=m_test,
        p_list=p_list, r_list=r_list,
        block_layout_fn=block_layout, fname="rom_runtime_vs_error.pdf",)

    for p, data in summary.items():
        print(f"p = {p}: r* = {data['r_star']}, "
              f"C_off = {data['C_off']:.4e}, "
              f"t_fom = {data['t_fom']:.4e}, "
              f"t_r = {data['t_r']:.4e}, "
              f"N_BE = {data['N_BE']:.4e}")
    return results_rt, summary

if __name__ == "__main__":
    main_exercise_3_2()