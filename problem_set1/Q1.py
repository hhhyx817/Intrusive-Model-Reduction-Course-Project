from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.sparse.linalg import cg
jax.config.update("jax_enable_x64", True)

# TODO 减少 for 循环, 减少时间复杂度

'''
Description in my remark:
1.  "grid" or "element" is the same refering the 1/n *1/n square, 
    "block" is one of the small thermal block containing several grids/elements
    finite element (gird + node basis function: take the 1 value only at this node, otherwise take 0 value)
    
1.  To keep the full-model implementation simple, assign each finite element to the thermal block containing its midpoint. 
    We ingore the error from the inconsistent conductivity on the block interface
'''

# -------------------------------------------------------------------------
# 0. Class Definition
# -------------------------------------------------------------------------
@dataclass
class Mesh:
    n: int
    h: float
    N: int
    node_coords: Array
    elem_nodes: Array
    free_nodes: Array
    node_to_dof: Array

@dataclass
class BlockMap:
    p: int
    Bx: int
    By: int
    elem_block: Array

@dataclass
class Model:
    mesh: Mesh
    block_map: BlockMap
    F: Array
    q: Array
    K_loc: Array
    M_loc: Array

'''
construct the grid
'''
# -------------------------------------------------------------------------
# 1. 主函数
# -------------------------------------------------------------------------
def build_mesh(l: int) -> Mesh:
    n = 2 ** l
    h = 1.0 / n
    N = n * (n + 1) # the number of the vertices removing the Dirichlect boundary

    # Node numbering 
    xs = jnp.arange(n + 1) * h
    ys = jnp.arange(n + 1) * h
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    node_coords = jnp.stack([X, Y], axis=-1)
    
    # Grid Vertices Numbering: every of n*n small grid has 4 vertices, we record them couter-clockwise
    elem_nodes = [] 
    for i in range(n):                                        # i: row index
        for j in range(n):
            n0 = node_index(i, j, n)                          # 3----2
            n1 = node_index(i + 1, j, n)                      # |    |
            n2 = node_index(i + 1, j + 1, n)                  # 0----1
            n3 = node_index(i, j + 1, n)
            elem_nodes.append([n0, n1, n2, n3])               # elem_nodes[4n-4] to elem_nodes[4n] for the block n
    elem_nodes = jnp.array(elem_nodes, dtype=jnp.int32)

    # Degree of Freedom numbering: Node numbering excluding the Dirichlet boundary
    node_to_dof = -jnp.ones((n + 1, n + 1), dtype=jnp.int32)
    counter = 0
    for i in range(n + 1):
        for j in range(n + 1):
            if not is_dirichlet(i, j, n):
                node_to_dof = node_to_dof.at[i, j].set(counter)
                counter += 1

    free_nodes = jnp.arange(N, dtype=jnp.int32)

    return Mesh(
        n=n,
        h=h,
        N=N,
        node_coords=node_coords,
        elem_nodes=elem_nodes,
        free_nodes=free_nodes,
        node_to_dof=node_to_dof,)


def node_index(i: int, j: int, n: int) -> int:
    # turn 2-dim (i,j) to 1-dim index i*(n+1)+j  n+1 node in vevey row, i/j in {0,1,..,n}
    return i* (n + 1) + j

def is_dirichlet(i: int, j: int, n: int) -> bool:
    return (j==0)

# -------------------------------------------------------------------------
# 2.主函数
# -------------------------------------------------------------------------
def build_block_map(mesh: Mesh, Bx: int, By: int) -> BlockMap:
    # determine which block the grid belongs to by using the center point
    midpoints = elem_midpoints(mesh)
    elem_block = locate_block(midpoints, Bx, By)
    
    # Bx blocks in the x direction, By blocks in the y direction
    return BlockMap(
        p=Bx * By,
        Bx=Bx,
        By=By,
        elem_block=elem_block,
    )


def locate_block(midpoints: Array, Bx: int, By: int) -> Array:
    x1 = midpoints[:, 0]
    x2 = midpoints[:, 1]
    
    bx = jnp.clip((x1 * Bx).astype(jnp.int32), 0, Bx - 1)
    by = jnp.clip((x2 * By).astype(jnp.int32), 0, By - 1)
    
    q = bx * By + by
    return q

def elem_midpoints(mesh: Mesh) -> Array:
    # find the center of every grid
    '''
    Tite statement: Assign each finite element to the thermal block that contains its center point.
    '''
    n = mesh.n
    h = mesh.h
    
    i_idx, j_idx = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    x1 = (i_idx + 0.5) * h
    x2 = (j_idx + 0.5) * h
    
    midpoints = jnp.stack([x1, x2], axis=-1).reshape(-1, 2)
    return midpoints

# -------------------------------------------------------------------------
# 3. 主函数
# -------------------------------------------------------------------------
# pack the blocks and grids, F(q), Mloc, Kloc to the model 
def build_model(ell, Bx, By):
    '''
    model = build_model(ell=5, Bx=3, By=3)
    print("F shape =", model.F.shape)                  # (1056,)
    print("F nonzero =", jnp.count_nonzero(model.F))   # 33
    print("F sum =", jnp.sum(model.F))                 # 1.0
    print("q is F:", jnp.allclose(model.q, model.F))   # True
    '''
    mesh = build_mesh(ell)
    block_map = build_block_map(mesh, Bx, By)
    K_loc = local_stiffness(mesh.h)
    M_loc = local_mass(mesh.h)

    # temporary Model, used to calculate F
    model_tmp = Model(
        mesh=mesh,
        block_map=block_map,
        F=jnp.zeros(mesh.N),
        q=jnp.zeros(mesh.N),
        K_loc=K_loc,
        M_loc=M_loc,
    )
    F = assemble_load(model_tmp)
    q = F 

    model = Model(
        mesh=mesh,
        block_map=block_map,
        F=F,
        q=q,
        K_loc=K_loc,
        M_loc=M_loc,
    )
    return model


def local_stiffness(h: float) -> Array:
    K = (1.0 / 6.0) * jnp.array([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0],
    ], dtype=jnp.float64)
    return K


def local_mass(h: float) -> Array:
    M = (h ** 2 / 36.0) * jnp.array([
        [4.0, 2.0, 1.0, 2.0],
        [2.0, 4.0, 2.0, 1.0],
        [1.0, 2.0, 4.0, 2.0],
        [2.0, 1.0, 2.0, 4.0],
    ], dtype=jnp.float64)
    return M

# calculate exactly the "Fi" in the orogin FOM F, i.e. the integral of ϕi on the Nueman boundary
def assemble_load(model: Model) -> Array:
    mesh = model.mesh
    n = mesh.n
    h = mesh.h
    N = mesh.N
    F = jnp.zeros(N, dtype=jnp.float64)

    # itern all the nodes on: Gamma_N = {x2 = 1}, i.e. j = n
    for i in range(n + 1): 
        k = int(mesh.node_to_dof[i, n])             # 1 |*                   /  \
        if k < 0:                                   #   | \                 /    \
            continue                                #   |  \               /      \
        if i == 0 or i == n:                        # 0 |---*---*        *---*---*---*
            F = F.at[k].set(h / 2.0)                #   x0  x1  x2       x0  x1  x2  x3
        else:
            F = F.at[k].set(h)
    return F
# q is defined as F
def assemble_qoi_weights(model: Model) -> Array:
    return model.F

# Map parameter dimension p to block layout (Bx, By).  p = Bx * By,
def block_layout(p: int) -> tuple:
    if p == 2: return 2, 1
    if p == 4: return 2, 2
    if p == 9: return 3, 3
    if p == 16: return 4, 4
    if p == 25: return 5, 5
    raise ValueError(f"Unsupported p: {p}")
# -------------------------------------------------------------------------
# 4. Matrix-Free Operator: calculate A(miu)·v for given v, which we use to get the final solution u
# -------------------------------------------------------------------------
"""
Apply the mass operator M.
Math: M v = sum_K M_loc v_K
"""
def apply_M(model: Model, v: Array) -> Array:

    mesh = model.mesh
    M_loc = model.M_loc
    elem_nodes = mesh.elem_nodes
    node_to_dof = mesh.node_to_dof

    v_nodes = _expand_to_nodes(mesh, v)     # Expand DOF vector to node values
    v_K_all = v_nodes[elem_nodes]           # Gather local node values (n^2, 4)    
    w_K_all = v_K_all @ M_loc.T             # Local mass matrix action (n^2, 4)

    # Scatter-add into global vector
    out = jnp.zeros(mesh.N, dtype=v.dtype)
    out = scatter_add(out, elem_nodes, w_K_all, node_to_dof)
    return out


def apply_local(K_loc: Array, v_K: Array) -> Array:
    """
    Apply the local matrix to the local vector.
    Math: w_K = K_loc @ v_K
    """
    return K_loc @ v_K


def scatter_add(v_global: Array,
                elem_nodes: Array,
                v_local: Array,
                node_to_dof: Array) -> Array:
    """
    Scatter-add local contributions into the global vector. There are 3 steps:
        1. Map global node indices to DOF indices via node_to_dof.
        2. Discard contributions from Dirichlet nodes (DOF index = -1).
        3. Accumulate local values into the global vector.

    Math: out[dof] += v_local[elem, a]
    """
    # Map global node indices to DOF indices
    elem_dofs = node_to_dof.reshape(-1)[elem_nodes]   # (n^2, 4)

    # Flatten for scatter-add
    dofs = elem_dofs.reshape(-1)                      # (4 n^2,)
    vals = v_local.reshape(-1)                        # (4 n^2,)

    # Mask out Dirichlet contributions
    mask = dofs >= 0
    dofs = jnp.where(mask, dofs, 0)
    vals = jnp.where(mask, vals, 0.0)

    v_global = v_global.at[dofs].add(vals)
    return v_global


# Expand a DOF vector to a full node vector, i.e. to add Dirichlet nodes
def _expand_to_nodes(mesh, v: Array) -> Array:
    """
    - Non-Dirichlet nodes get the corresponding DOF value.
    - Dirichlet nodes remain 0.

    Math: v_nodes[node] = v[dof] if node is free, else 0
    """
    node_to_dof_flat = mesh.node_to_dof.reshape(-1)   # (M,), M = (n+1)^2

    # Replace -1 (Dirichlet) with 0 to make a safe integer index array.
    safe_indices = jnp.where(node_to_dof_flat >= 0, node_to_dof_flat, 0)  # (M,)

    # Gather values: for non-Dirichlet nodes this is v[dof]; for Dirichlet nodes it is v[0].
    values = v[safe_indices]                          # (M,)

    # Zero out Dirichlet positions.
    values = jnp.where(node_to_dof_flat >= 0, values, 0.0)  # (M,)

    return values


def apply_Aq(model: Model, q_idx: int, v: Array) -> Array:
    """
    Apply a single affine component A_q.
    Math: (A_q v) = sum_{K in Omega_q} K_loc v_K
    """
    mesh = model.mesh
    block_map = model.block_map
    K_loc = model.K_loc

    elem_nodes = mesh.elem_nodes            # (n^2, 4)
    elem_block = block_map.elem_block       # (n^2,)
    node_to_dof = mesh.node_to_dof

    # Expand DOF vector to node values
    v_nodes = _expand_to_nodes(mesh, v)
    v_K_all = v_nodes[elem_nodes]           # Gather local node values for each element (n^2, 4)

    # Keep only elements belonging to block q_idx
    mask_q = (elem_block == q_idx)
    v_K_all = jnp.where(mask_q[:, None], v_K_all, 0.0)
    w_K_all = v_K_all @ K_loc.T             #  Local matrix action (n^2, 4)

    # Scatter-add into global vector
    out = jnp.zeros(mesh.N, dtype=v.dtype)
    out = scatter_add(out, elem_nodes, w_K_all, node_to_dof)
    return out


def apply_A(model: Model, mu: Array, v: Array) -> Array:
    """
    Apply the full affine operator A(mu).
    Math: A(mu) v = sum_q mu_q A_q v
                  = sum_K mu_{q(K)} K_loc v_K
    """
    mesh = model.mesh
    block_map = model.block_map
    K_loc = model.K_loc

    elem_nodes = mesh.elem_nodes
    elem_block = block_map.elem_block
    node_to_dof = mesh.node_to_dof

    # Expand DOF vector to node values
    v_nodes = _expand_to_nodes(mesh, v)    
    v_K_all = v_nodes[elem_nodes]           # Gather local node values for each element (n^2, 4)   
    mu_K = mu[elem_block]                   # Multiply by the block coefficient mu_{q(K)} (n^2,)
    v_K_all = v_K_all * mu_K[:, None]
    w_K_all = v_K_all @ K_loc.T             # Local matrix action (n^2, 4)

    # Scatter-add into global vector
    out = jnp.zeros(mesh.N, dtype=v.dtype)
    out = scatter_add(out, elem_nodes, w_K_all, node_to_dof)
    return out


# -------------------------------------------------------------------------
# 5. Other helping function: used later
# -------------------------------------------------------------------------
def jacobi_diagonal(model: Model, mu: Array) -> Array:
    """
    Compute the diagonal of A(mu) in a matrix-free way.
    Math: D_i = A(mu)_{ii}
              = sum_q mu_q (A_q)_{ii}
              = sum_K mu_{q(K)} (K_loc)_{aa}
    """
    mesh = model.mesh
    block_map = model.block_map
    K_loc = model.K_loc

    elem_nodes = mesh.elem_nodes            # (n^2, 4)
    elem_block = block_map.elem_block       # (n^2,)
    node_to_dof = mesh.node_to_dof

    n_elems = elem_nodes.shape[0]
    D = jnp.zeros(mesh.N, dtype=jnp.float64)

    # Diagonal entries of the local stiffness matrix (4,)
    K_diag = jnp.diag(K_loc)               
    # Coefficient per element (n^2,)
    mu_K = mu[elem_block]                   
    # Contribution per element: mu_{q(K)} * K_diag[a]  (n^2, 4)
    contrib = mu_K[:, None] * K_diag[None, :]   
    # Map node indices to DOF indices  (n^2, 4)
    elem_dofs = node_to_dof.reshape(-1)[elem_nodes]   

    dofs = elem_dofs.reshape(-1)            # (4 n^2,)
    vals = contrib.reshape(-1)              # (4 n^2,)

    mask = dofs >= 0
    dofs = jnp.where(mask, dofs, 0)
    vals = jnp.where(mask, vals, 0.0)

    D = D.at[dofs].add(vals)
    return D


def make_preconditioner(model: Model, mu: Array) -> Callable[[Array], Array]:
    """
    Build a Jacobi preconditioner.
    Math: M^{-1} v = D^{-1} v
    """
    D = jacobi_diagonal(model, mu)
    eps = 1e-14
    D_safe = jnp.where(jnp.abs(D) < eps, 1.0, D) # Avoid division by zero

    def precond(v: Array) -> Array:
        return v / D_safe
    return precond


def solve(model: Model, mu: Array) -> Array:
    """
    Solve A(mu) u = F using matrix-free CG.
    Math: A(mu) u = F
          A(mu)   = sum_q mu_q A_q
    """
    def matvec(v: Array) -> Array:
        return apply_A(model, mu, v)

    M = make_preconditioner(model, mu)

    u, info = cg(
        matvec,
        model.F,
        M=M,
        tol=1e-10,
        maxiter=2000,
    )
    return u


def qoi(model: Model, mu: Array) -> Array:
    """
    Compute the quantity of interest.
    Math: Q(mu) = q^T u(mu) = F^T u(mu)
    """
    u = solve(model, mu)
    return model.q @ u


# -------------------------------------------------------------------------
# 6. Solution fo Question1 
# -------------------------------------------------------------------------
def draw_params(key: Array, m: int, p: int) -> Array:
    """
    Draw m samples of the parameter vector miu in R^p. miu is the heat conductivity coafficient.
    Math: miu^(i) ~ U([a, b]^p),  i = 1, ..., m
          returns Miu of shape (m, p)
    """
    a, b = 0.1, 10.0
    key, subkey = jax.random.split(key) # repeatable
    Mu = jax.random.uniform(
        subkey,
        shape=(m, p),
        minval=a,
        maxval=b,
        dtype=jnp.float64,
    )
    return Mu


def snapshots(model: Model, mus: Array) -> Array:
    """
    Compute snapshot matrix S = [u(mu_1), ..., u(mu_m)].
    Math: For each i, solve A(mu_i) u_i = F,
          then stack u_i as columns of S.
    """
    
# Compute snapshot matrix S = [u(mu_1), ..., u(mu_m)].
def snapshots(model: Model, mus: Array) -> Array:
    """
    Math: For each i, solve A(mu_i) u_i = F,
          then stack u_i as columns of S.
    """
    """
    m = mus.shape[0]
    cols = [solve(model, mus[i]) for i in range(m)]
    S = jnp.stack(cols, axis=1)   # (N, m)
    return S
    """
    # 使用 jax.vmap 将 solve 自动向量化，vmap 沿 mus 的第一个轴（axis=0）进行映射
    S = jax.vmap(lambda mu: solve(model, mu))(mus)  # (m, N)
    return S.T


def exact_solution(x: Array, c: float) -> float:
    """
    Exact solution for the constant-coefficient case mu_q = c for all q.
    Math: u(x) = x_2 / c
    """
    return x[1] / c


def exact_qoi(c: float) -> float:
    """
    Exact QoI for the constant-coefficient case.
    Math: Q(c) = integral_{Gamma_N} u ds = 1 / c
    """
    return 1.0 / c


def test_exact_solution(p: int, ell: int, c: float) -> None:
    """
    Verify the FOM solver against the exact solution for miu_q = c.
    Math:
        miu = c * ones(p)
        u_h = A(mu)^{-1} F
        u_exact(x) = x_2 / c
        Q_h = F^T u_h,  Q_exact = 1 / c
    """
    # Build model with p = Bx * By blocks, Choose Bx, By such that Bx * By = p
    Bx = int(round(jnp.sqrt(p)))
    while p % Bx != 0:
        Bx -= 1
    By = p // Bx

    model = build_model(ell, Bx, By)

    # Constant parameter
    mu = c * jnp.ones(p, dtype=jnp.float64)

    # FOM solution
    u_h = solve(model, mu)

    # Exact solution at free nodes
    node_coords = model.mesh.node_coords.reshape(-1, 2)   # (N_nodes, 2)
    node_to_dof = model.mesh.node_to_dof.reshape(-1)      # (N_nodes,)
    mask = node_to_dof >= 0

    x2_free = node_coords[mask, 1]                        # (N,)
    u_exact = x2_free / c

    err_u = jnp.linalg.norm(u_h - u_exact) / jnp.linalg.norm(u_exact)

    Q_h = model.q @ u_h
    Q_exact = exact_qoi(c)
    err_Q = jnp.abs(Q_h - Q_exact) / jnp.abs(Q_exact)

    print(f"[test_exact_solution] p={p}, ell={ell}, c={c}")
    print(f"  relative error u   = {err_u:.3e}")
    print(f"  Q_h                = {Q_h:.6f}")
    print(f"  Q_exact            = {Q_exact:.6f}")
    print(f"  relative error Q   = {err_Q:.3e}")

    
def main():
    print("Exercise 1.1 skeleton is ready.")
    # Quick sanity check
    for p_test in [2, 4, 9, 16, 25]:
        test_exact_solution(p=p_test, ell=4, c=2.0)

if __name__ == "__main__":
    main()