'''
【Overview】
The goal is to understand how the singular value spectrum of the snapshot matrix depends on
    (a) the parameter dimension p, --- 2.1
    (b) the mesh level ell.        --- 2.2
 

[2.1] Snapshot spectra for increasing the number of thermal blocks p
    fix the mesh level ell = 5 and vary the parameter dimension p in {2, 4, 9, 16, 25}.
    For each p:
    - draw m >= 1000 independent parameters from Unif([0.1, 10]^p),
    - generate the snapshot matrix U_p in R^{N x m},
    - compute the singular values sigma^{(p)}_j,
    - compute the normalized spectrum sigma^{(p)}_j / sigma^{(p)}_1,
    - plot all normalized spectra in one semilogarithmic figure,
    - report the smallest r s.t. (sum_{j > r} sigma_j^2) / (sum_j sigma_j^2) <= 1e-4.

R can be the dimension of the reduced subspace. This shows how the effective rank grows with the parameter dimension.


[2.2] Snapshot spectra under mesh refinement
    Fix p = 9. Draw one set of m = 1000 parameters, and reuse exactly the same parameters for each mesh level 'ell' in {4, 5, 6}.
    For each ell:
    - draw m >= 1000 independent parameters from Unif([0.1, 10]^p), reuse exactly the same parameters
    - generate the snapshot matrix U_ell in R^{N_ell x m},
    - compute the singular values sigma^{(ell)}_j,
    - compute the normalized spectrum sigma^{(ell)}_j / sigma^{(ell)}_1,
    - plot all normalized spectra in one semilogarithmic figure.
    
Rapid decay of the singular values indicates that the solution manifold is low-dimensional and therefore reducible by POD.
'''

from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.sparse.linalg import cg
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from Q1 import (build_model, draw_params, snapshots, block_layout,)


# ------------------------------------------------------------
# Singular value analysis / visualization
# ------------------------------------------------------------
def compute_svd_eigen(U: Array) -> Array:
    """
    Compute the singular values of the snapshot matrix U, i.e. sigma_1 >= sigma_2 >= ... >= 0 with U = W Sigma V^T
    U : (N, m) snapshot matrix --> sigma : (min(N, m),) array of singular values
    """
    sigma = jnp.linalg.svd(U, full_matrices=False, compute_uv=False)
    return sigma

def normalized_spectrum(sigma: Array) -> Array:
    # Normalize singular value spectrum sigma_hat_j = sigma_j / sigma_1
    return sigma / sigma[0]  

def effective_rank(sigma: Array, tol: float) -> int:
    """
    Find the smallest r such that the tail energy is below tol(energy tolerance).
    (sum_{j > r} sigma_j^2) / (sum_j sigma_j^2) <= tol
    """
    energy = sigma ** 2
    total = jnp.sum(energy)
    tail = jnp.cumsum(energy[::-1])[::-1]   # tail[k] = sum_{j >= k} energy[j]
    ratio = tail / total
    
    k = int(jnp.argmax(ratio <= tol))
    r_min = max(k - 1, 0)
    return r_min


def plot_spectra(sigmas: list, labels: list, title: str, fname: str) -> None:
    """
    Plot normalized singular value spectra on a semilogarithmic axis.
        y-axis : sigma_hat_j = sigma_j / sigma_1
        x-axis : index j
    """
    plt.figure(figsize=(7, 5))
    for sigma_hat, label in zip(sigmas, labels):
        j = jnp.arange(1, len(sigma_hat) + 1)
        plt.semilogy(j, sigma_hat, label=label)
    plt.xlabel("index j")
    plt.ylabel(r"$\sigma_j / \sigma_1$")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    

# ------------------------------------------------------------
#  Driver
# ------------------------------------------------------------
# 1. Exercise 1.2.1: snapshot spectra for increasing the number of thermal blocks p
def run_exercise_2_1(ell: int, m: int, p_list: list, block_layout_fn: Callable, energy_tol: float, fname: str) -> dict:
    """
    ell            : mesh level (e.g. 5)
    m              : number of parameter samples (>= 1000)
    p_list         : list of parameter dimensions (number of thermal blocks), e.g. [2,4,9,16,25]
    block_layout_fn: function p -> (Bx, By)
    energy_tol     : energy tolerance (e.g. 1e-4)
    fname          : output file name for the plot
    results        : dict mapping p -> r_p
    """
    sigmas = []
    labels = []
    results = {}

    key = jax.random.key(0)

    for p in p_list:
        Bx, By = block_layout_fn(p)
        model = build_model(ell, Bx, By)

        key, subkey = jax.random.split(key)
        mus = draw_params(subkey, m, p)
        U = snapshots(model, mus)

        sigma = compute_svd_eigen(U)
        sigma_hat = normalized_spectrum(sigma)
        r = effective_rank(sigma, energy_tol)

        sigmas.append(sigma_hat)
        labels.append(f"p = {p}")
        results[p] = r
        print(f"p = {p:2d}, smallest r with tail <= {energy_tol}: {r}")

    plot_spectra(sigmas, labels,
                 title=f"Normalized snapshot spectra (ell = {ell})",
                 fname=fname)
    return results


# 2. Exercise 1.2.2: snapshot spectra under mesh refinement with incresaing ell
def run_exercise_2_2(p: int, m: int, ell_list: list, block_layout_fn: Callable, fname: str) -> None:
    """
    p              : parameter dimension (number of thermal blocks), e.g. 9
    m              : number of parameter samples (e.g. 1000)
    ell_list       : list of mesh levels, e.g. [4, 5, 6]
    block_layout_fn: function p -> (Bx, By)
    fname          : output file name for the plot
    """
    Bx, By = block_layout_fn(p)
    key = jax.random.key(0)
    mus = draw_params(key, m, p)     # same parameters for all ell

    sigmas = []
    labels = []

    for ell in ell_list:
        model = build_model(ell, Bx, By)
        U = snapshots(model, mus)

        sigma = compute_svd_eigen(U)
        sigma_hat = normalized_spectrum(sigma)
        sigmas.append(sigma_hat)
        labels.append(f"ell = {ell}")

    plot_spectra(sigmas, labels,
                 title=f"Normalized snapshot spectra (p = {p})",
                 fname=fname)


def main_exercise_1and2():
    """
    1. Run run_exercise_2_1 with ell=5, m>=1000, p_list=[2,4,9,16,25].
    2. Run run_exercise_2_2 with p=9, m=1000, ell_list=[4,5,6].
    """
    ell = 5
    m = 1000
    p_list = [2, 4, 9, 16, 25]
    energy_tol = 1e-4

    print("Exercise 1.2.1: snapshot spectra for increasing p")
    results = run_exercise_2_1(
        ell=ell,
        m=m,
        p_list=p_list,
        block_layout_fn=block_layout,
        energy_tol=energy_tol,
        fname="spectra_p.pdf",)
    print("Effective ranks:", results)

    print("Exercise 1.2.2: snapshot spectra under mesh refinement")
    run_exercise_2_2(
        p=9,
        m=m,
        ell_list=[4, 5, 6],
        block_layout_fn=block_layout,
        fname="spectra_ell.pdf",)
    

if __name__ == "__main__":
    main_exercise_1and2()




