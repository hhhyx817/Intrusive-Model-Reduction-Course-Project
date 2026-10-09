# Intrusive and Non-Intrusive Model Reduction

**Homework 1 — Thermal Block: Full Model, Reduction, and UQ**

*EPFL — Intrusive and Non-Intrusive Model Reduction — Fall 2026*

---

## Overview

This repository contains the code, data, and report for Homework 1 on
**model reduction of a parametrized thermal block problem**. The goal is to
build a matrix-free full-order model (FOM) in JAX, study its reducibility,
construct a POD–Galerkin reduced-order model (ROM), and use the ROM as a
cheap surrogate for uncertainty quantification (UQ) and multi-fidelity
estimation.

All heavy computation is done in **JAX**, with **JIT compilation**, **double
precision**, and **matrix-free** operators. The full-order stiffness matrix
is never assembled as a dense or sparse matrix.

---

## Problem Background

### The continuous problem

Let $\Omega = (0,1)^2$ be the unit square, partitioned into a boundary

- $\Gamma_D$ — Dirichlet boundary (zero temperature),
- $\Gamma_N$ — Neumann boundary (unit heat flux),
- and the remaining boundary (zero flux).

We solve the stationary heat equation

$$
-\nabla \cdot \bigl(\kappa(x;\mu)\,\nabla u(x;\mu)\bigr) = 0 \quad \text{in } \Omega,
$$

with boundary conditions

$$
u = 0 \ \text{on } \Gamma_D, \qquad
\kappa \nabla u \cdot n = 1 \ \text{on } \Gamma_N, \qquad
\kappa \nabla u \cdot n = 0 \ \text{elsewhere}.
$$

### Parametrization: the thermal block

The conductivity field is piecewise constant on a $B_x \times B_y$ partition
of $\Omega$:

$$
\kappa(x;\mu) = \sum_{q=1}^{p} \mu_q \, \chi_{\Omega_q}(x),
\qquad \mu \in \mathcal{D}_p = [0.1, 10]^p.
$$

We consider the parameter dimensions

| $p$ | Block layout $(B_x, B_y)$ |
|:---:|:-------------------------:|
| 2   | (2, 1)                    |
| 4   | (2, 2)                    |
| 9   | (3, 3)                    |
| 16  | (4, 4)                    |
| 25  | (5, 5)                    |

The default case is $p = 9$.

### Discretization

At refinement level $\ell$, the domain is meshed with $n = 2^\ell$
rectangular elements per coordinate direction, with mesh width $h = 1/n$.
The finite-element space uses tensor products of 1D hat functions
$\phi_{ij}(x_1,x_2) = \ell_i(x_1) m_j(x_2)$, with the basis functions on
$\Gamma_D$ removed. The number of unknowns is

$$
N = n(n+1) = 2^\ell(2^\ell+1).
$$

The discrete operator is affine in the parameters:

$$
A(\mu) = \sum_{q=1}^{p} \mu_q \, A_q,
$$

but the matrices $A_q$ are **never assembled**. Instead, the code stores the
element connectivity and the element-to-block map, and provides matrix-free
applications of $A(\mu)$, $A_q$, and the mass matrix $M$.

### Quantity of interest

The quantity of interest (QoI) is the average temperature on the heated edge:

$$
Q(\mu) = \int_{\Gamma_N} u_N(x;\mu)\, ds = q^T u_N(\mu),
$$

where $q = F$ because the same boundary functional defines the load vector.

---

## Repository Structure
.
├── README.md 
├── report.pdf 
├── src/
│ ├── model.py # mesh, connectivity, element-to-block map
│ ├── operators.py # matrix-free A(μ)v, A_q v, Mv
│ ├── solve.py # JIT-compiled CG solve and QoI
│ ├── snapshots.py # snapshot generation
│ ├── pod.py # Euclidean POD and Galerkin ROM
│ ├── timings.py # runtime and speedup measurements
│ └── uq.py # Monte Carlo, control variates
├── scripts/
│ ├── run_ex1_1.py # FOM validation
│ ├── run_ex1_2.py # snapshot spectra
│ ├── run_ex1_3.py # POD / ROM errors and timings
│ └── run_ex1_4.py # UQ and multi-fidelity
├── data/ # raw numerical data for figures
├── figures/ # generated plots
└── environment.yml # JAX + dependencies