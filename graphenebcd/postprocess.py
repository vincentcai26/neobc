#!/usr/bin/env python3
"""
Post-process parsed real-space matrices and build H(k).

Usage examples:
  python postprocess.py --k-frac 0.0 0.0 0.0
  python postprocess.py --k-cart 0.10 0.20 0.00
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HARTREE_TO_EV = 27.2113845
SEMENOFF_MASS_EV = 0
BASIS_INDICES_FILE = "basis-indices.out"


def read_lattice_vectors(geometry_path: Path) -> np.ndarray:
    lattice = []
    with geometry_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0] == "lattice_vector":
                lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(lattice) != 3:
        raise ValueError("Expected exactly 3 lattice_vector lines in geometry.in")
    return np.array(lattice, dtype=float)


def reciprocal_lattice(a: np.ndarray) -> np.ndarray:
    # a has direct vectors as rows (A)
    # B rows are reciprocal vectors (1/Angstrom), with A @ B^T = 2*pi*I
    return 2.0 * np.pi * np.linalg.inv(a).T


def cell_indices_to_cartesian(cell_indices: np.ndarray, a: np.ndarray) -> np.ndarray:
    # R = n1*a1 + n2*a2 + n3*a3
    return cell_indices @ a


def build_k_matrix(mat_r: np.ndarray, r_cart: np.ndarray, k_cart: np.ndarray) -> np.ndarray:
    # H(k) = sum_R exp(i k·R) H(R)
    phases = np.exp(1j * (r_cart @ k_cart))
    return np.tensordot(phases, mat_r, axes=(0, 0))


def parse_sparse_indices(indices_path: Path):
    """
    Parse rs_indices.out preserving raw indexing as written by FHI-aims.
    """
    lines = [ln.strip() for ln in indices_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    i = 0

    def parse_tag(expected: str) -> int:
        nonlocal i
        tag, val = lines[i].split(":", 1)
        if tag.strip() != expected:
            raise ValueError(f"Expected '{expected}', got '{tag.strip()}'")
        i += 1
        return int(val.strip())

    n_values = parse_tag("n_hamiltonian_matrix_size")
    n_cells_raw = parse_tag("n_cells_in_hamiltonian")
    n_basis = parse_tag("n_basis")

    if lines[i] != "cell_index":
        raise ValueError(f"Expected cell_index block, got {lines[i]}")
    i += 1
    cell_index = np.zeros((n_cells_raw, 3), dtype=int)
    for c in range(n_cells_raw):
        cell_index[c] = [int(x) for x in lines[i].split()]
        i += 1

    if lines[i] != "index_hamiltonian(1,:,:)":
        raise ValueError(f"Expected index_hamiltonian(1,:,:), got {lines[i]}")
    i += 1
    idx_start = np.zeros((n_cells_raw, n_basis), dtype=int)
    for c in range(n_cells_raw):
        idx_start[c] = [int(x) for x in lines[i].split()]
        i += 1

    if lines[i] != "index_hamiltonian(2,:,:)":
        raise ValueError(f"Expected index_hamiltonian(2,:,:), got {lines[i]}")
    i += 1
    idx_end = np.zeros((n_cells_raw, n_basis), dtype=int)
    for c in range(n_cells_raw):
        idx_end[c] = [int(x) for x in lines[i].split()]
        i += 1

    if lines[i] != "column_index_hamiltonian":
        raise ValueError(f"Expected column_index_hamiltonian, got {lines[i]}")
    i += 1
    col_index = np.array([int(x) for x in lines[i:]], dtype=int)

    if len(col_index) != n_values:
        raise ValueError(
            f"column_index length {len(col_index)} != n_hamiltonian_matrix_size {n_values}"
        )

    return n_values, n_cells_raw, n_basis, cell_index, idx_start, idx_end, col_index


def read_sparse_values(path: Path, n_expected: int) -> np.ndarray:
    vals = np.array(
        [float(ln.strip()) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()],
        dtype=float,
    )
    if len(vals) != n_expected:
        raise ValueError(f"{path} has {len(vals)} values, expected {n_expected}")
    return vals


def build_sparse_map(
    n_cells_raw: int,
    n_basis: int,
    idx_start: np.ndarray,
    idx_end: np.ndarray,
    col_index: np.ndarray,
):
    """
    Precompute sparse entry maps for fast k-matrix reconstruction.
    """
    row = []
    col = []
    cell = []
    place = []
    mirror = []

    # Mirror FHI-aims loop: i_cell = 1..n_cells_in_hamiltonian-1
    for c in range(n_cells_raw - 1):
        for i_basis in range(n_basis):
            start = idx_start[c, i_basis]
            end = idx_end[c, i_basis]
            if start <= 0:
                continue
            for p in range(start, end + 1):  # Fortran inclusive range
                j_basis = col_index[p - 1] - 1
                row.append(i_basis)
                col.append(j_basis)
                cell.append(c)
                place.append(p - 1)  # 0-based into sparse_values
                mirror.append(i_basis != j_basis)

    return {
        "row": np.array(row, dtype=int),
        "col": np.array(col, dtype=int),
        "cell": np.array(cell, dtype=int),
        "place": np.array(place, dtype=int),
        "mirror": np.array(mirror, dtype=bool),
    }


def build_k_matrix_from_sparse(
    sparse_values: np.ndarray,
    k_cart: np.ndarray,
    a: np.ndarray,
    cell_index: np.ndarray,
    sparse_map,
) -> np.ndarray:
    """
    Reconstruct k-dependent dense matrix from FHI-aims sparse real-space format.
    Mirrors FHI-aims construct_dense_matrix_from_sparse:
      M(i,j) += conj(k_phase(cell)) * sparse(place)
      M(j,i) +=      k_phase(cell)  * sparse(place), for i != j
    and loops cells up to n_cells_in_hamiltonian-1 (exclude sentinel last row).
    """
    n_basis = int(np.max([sparse_map["row"].max(), sparse_map["col"].max()])) + 1
    mat = np.zeros((n_basis, n_basis), dtype=complex)

    # Phases for all non-sentinel cells
    r_cart_cells = cell_index[:-1] @ a
    phases = np.exp(1j * (r_cart_cells @ k_cart))

    row = sparse_map["row"]
    col = sparse_map["col"]
    cell = sparse_map["cell"]
    place = sparse_map["place"]
    mirror = sparse_map["mirror"]

    vals = sparse_values[place]

    # M(i,j) += conj(phase(cell)) * sparse(place)
    contrib_ij = np.conj(phases[cell]) * vals
    np.add.at(mat, (row, col), contrib_ij)

    # M(j,i) += phase(cell) * sparse(place), for i != j
    if np.any(mirror):
        contrib_ji = phases[cell[mirror]] * vals[mirror]
        np.add.at(mat, (col[mirror], row[mirror]), contrib_ji)

    return mat


def clip_polygon_with_halfplane(
    poly: np.ndarray, normal: np.ndarray, bound: float, tol: float = 1e-12
) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        f0 = float(np.dot(normal, p0) - bound)
        f1 = float(np.dot(normal, p1) - bound)
        inside0 = f0 <= tol
        inside1 = f1 <= tol

        if inside0 and inside1:
            out.append(p1)
        elif inside0 and not inside1:
            t = f0 / (f0 - f1)
            out.append(p0 + t * (p1 - p0))
        elif (not inside0) and inside1:
            t = f0 / (f0 - f1)
            out.append(p0 + t * (p1 - p0))
            out.append(p1)

    if not out:
        return np.empty((0, 2), dtype=float)
    return np.array(out, dtype=float)


def first_bz_polygon_2d(b1: np.ndarray, b2: np.ndarray, shell: int = 2) -> np.ndarray:
    # Build 2D Wigner-Seitz cell in reciprocal space around Gamma:
    # keep k points satisfying k·G <= |G|^2 / 2 for all nearby nonzero G.
    radius = max(np.linalg.norm(b1), np.linalg.norm(b2)) * 4.0
    poly = np.array(
        [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]],
        dtype=float,
    )

    g_list = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if i == 0 and j == 0:
                continue
            g = i * b1 + j * b2
            g_list.append(g)

    g_list.sort(key=np.linalg.norm)
    for g in g_list:
        nvec = g
        bound = 0.5 * float(np.dot(g, g))
        poly = clip_polygon_with_halfplane(poly, nvec, bound)
        if len(poly) == 0:
            break
    return poly


def generate_kgrid(num_x,num_y):
    """
    Uniform 2D mesh in reciprocal *fractional* coordinates.
    Uses [-0.5, 0.5) to avoid duplicate periodic boundary points.

    Note: This is a centered primitive reciprocal cell mesh, not the
    Wigner-Seitz first-BZ polygon mesh.
    """
    kx_range = np.linspace(-0.5, 0.5, num_x, endpoint=False)
    ky_range = np.linspace(-0.5, 0.5, num_y, endpoint=False)
    X, Y = np.meshgrid(kx_range, ky_range)
    Z = np.zeros_like(X)
    grid_3d = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return grid_3d


def orthogonalize_hk(hk, sk):
    """
    Lowdin symmetric orthogonalization at one k-point.

    Given H(k), S(k), computes:
      S = U diag(lam) U^dagger
      X = U diag(lam^{-1/2}) U^dagger
      H_orth = X^dagger H X

    Returns:
      h_orth : orthogonalized Hamiltonian
      x      : orthogonalization matrix X
    """
    # Enforce Hermitian form against small numerical asymmetries
    hk = 0.5 * (hk + hk.conj().T)
    sk = 0.5 * (sk + sk.conj().T)

    # Diagonalize overlap matrix
    evals, evecs = np.linalg.eigh(sk)

    # Keep numerically positive subspace only
    tol = 1e-8
    keep = evals > tol
    if not np.any(keep):
        raise ValueError("S(k) has no eigenvalues above tolerance; cannot orthogonalize.")

    u = evecs[:, keep]
    lam_inv_sqrt = 1.0 / np.sqrt(evals[keep])

    # X = U diag(lam^{-1/2}) U^dagger in the retained subspace
    x = u @ np.diag(lam_inv_sqrt) @ u.conj().T

    # H' = X^dagger H X
    h_orth = x.conj().T @ hk @ x
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    return h_orth, x


def parse_basis_to_atom_map(path: Path, n_basis: int) -> np.ndarray:
    """
    Parse basis-indices.out and return atom index (1-based) for each basis function.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms = np.zeros(n_basis, dtype=int)
    n_found = 0

    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        if not parts[0].isdigit():
            continue
        bf = int(parts[0])  # 1-based basis function index
        at = int(parts[2])  # 1-based atom index
        if bf < 1 or bf > n_basis:
            continue
        atoms[bf - 1] = at
        n_found += 1

    if n_found != n_basis or np.any(atoms == 0):
        raise ValueError(
            f"Could not parse complete basis->atom map from {path}. "
            f"Found {n_found} basis entries, expected {n_basis}."
        )
    return atoms


def build_sublattice_signs(n_basis: int, basis_indices_path: Path) -> np.ndarray:
    """
    Build Semenoff A/B signs from explicit basis->atom mapping.
    For primitive graphene in this workflow: atom 1 = A (+1), atom 2 = B (-1).
    """
    basis_atom = parse_basis_to_atom_map(basis_indices_path, n_basis)
    unique_atoms = np.unique(basis_atom)
    if len(unique_atoms) != 2:
        raise ValueError(
            f"Expected exactly 2 atoms in basis map for primitive graphene, got {len(unique_atoms)}."
        )

    signs = np.where(basis_atom == unique_atoms[0], 1.0, -1.0)
    return signs


def add_semenoff_mass(hk: np.ndarray, mass_ev: float, sublattice_signs: np.ndarray) -> np.ndarray:
    """
    Add Semenoff term M * eta_i to onsite diagonal (Hartree units in hk).
    eta_i = +1 for A basis functions, -1 for B basis functions.
    """
    mass_hartree = mass_ev / HARTREE_TO_EV
    return hk + np.diag(mass_hartree * sublattice_signs)


def read_fermi_level_eV(aims_out_path: Path) -> float | None:
    """
    Read Fermi level (chemical potential) from aims.out.
    Returns None if not found.
    """
    patterns = [
        re.compile(r"Chemical potential \(Fermi level\):\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)"),
        re.compile(r"Chemical Potential\s*:\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*eV"),
    ]
    matches: list[float] = []
    try:
        with aims_out_path.open("r", encoding="utf-8") as f:
            for line in f:
                for pat in patterns:
                    m = pat.search(line)
                    if m:
                        matches.append(float(m.group(1)))
    except FileNotFoundError:
        return None
    if not matches:
        return None
    # Use the last reported value (typically final SCF report).
    return matches[-1]


# Step 1: Parse sparse rs_* matrices and lattice vectors.
geometry_file = "geometry.in"
a = read_lattice_vectors(Path(geometry_file))
b = reciprocal_lattice(a)

print(f"A (real space) lattice vectors: {a}")
print(f"B (reciprocal space) lattice vectors: {b}")

n_values, n_cells_raw, n_basis, cell_index, idx_start, idx_end, col_index = parse_sparse_indices(
    Path("rs_indices.out")
)
h_sparse = read_sparse_values(Path("rs_hamiltonian.out"), n_values)
s_sparse = read_sparse_values(Path("rs_overlap.out"), n_values)
sparse_map = build_sparse_map(n_cells_raw, n_basis, idx_start, idx_end, col_index)
sublattice_signs = build_sublattice_signs(n_basis, Path(BASIS_INDICES_FILE))

# Step 2: Generate k-points and build H(k) and S(k) matrices.
num_kx, num_ky = 51,51
k_points = generate_kgrid(num_kx, num_ky)
evals_all = []
for k_point in k_points:
    k_cart = k_point @ b
    hk = build_k_matrix_from_sparse(h_sparse, k_cart, a, cell_index, sparse_map)
    sk = build_k_matrix_from_sparse(s_sparse, k_cart, a, cell_index, sparse_map)
    hk = add_semenoff_mass(hk, SEMENOFF_MASS_EV, sublattice_signs)

    # Orthogonalize hk
    hk_orth, xk = orthogonalize_hk(hk, sk)

    # Eigenvalues
    evals = np.linalg.eigvalsh(hk_orth)
    evals_all.append(evals)

# Step 3: Heatmap for valence or conduction band on current k-mesh.
evals_all = np.array(evals_all)  # (num_kx*num_ky, n_bands)
evals_all_eV = evals_all * HARTREE_TO_EV
n_bands = evals_all.shape[1]

fermi_eV = read_fermi_level_eV(Path("aims.out"))
if fermi_eV is None:
    print("Warning: Could not find Fermi level in aims.out; using middle-band fallback.")

# Build valence and conduction surfaces first.
valence_vals_flat = np.full(evals_all.shape[0], np.nan, dtype=float)
conduction_vals_flat = np.full(evals_all.shape[0], np.nan, dtype=float)

if fermi_eV is not None:
    for i, ev in enumerate(evals_all_eV):
        below = ev[ev <= fermi_eV]
        above = ev[ev >= fermi_eV]
        if below.size > 0:
            valence_vals_flat[i] = below.max()
        if above.size > 0:
            conduction_vals_flat[i] = above.min()

# If EF-based classification fails (or EF missing), use middle-band fallback.
if np.all(np.isnan(valence_vals_flat)) or np.all(np.isnan(conduction_vals_flat)):
    print("Warning: EF-based valence/conduction selection failed; using middle-band fallback.")
    valence_idx = n_bands // 2 - 1
    conduction_idx = n_bands // 2
    valence_vals_flat = evals_all_eV[:, valence_idx]
    conduction_vals_flat = evals_all_eV[:, conduction_idx]

valence_vals = valence_vals_flat.reshape(num_ky, num_kx)
conduction_vals = conduction_vals_flat.reshape(num_ky, num_kx)
gap_vals = conduction_vals - valence_vals
k_cart_all = k_points @ b
kx = k_cart_all[:, 0].reshape(num_ky, num_kx)
ky = k_cart_all[:, 1].reshape(num_ky, num_kx)

# Gap heatmap
plt.figure(figsize=(6.5, 5.5))
mesh = plt.pcolormesh(kx, ky, gap_vals, shading="auto", cmap="coolwarm")
plt.colorbar(mesh, label=r"$E_c(k) - E_v(k)$ (eV)")
plt.xlabel(r"$k_x$ (1/$\AA$)")
plt.ylabel(r"$k_y$ (1/$\AA$)")
if fermi_eV is not None:
    plt.title(rf"Gap map $E_c(k)-E_v(k)$ (EF={fermi_eV:.4f} eV)")
else:
    plt.title(r"Gap map $E_c(k)-E_v(k)$")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("gap_heatmap.png", dpi=200)

# 3D valence/conduction surfaces
fig = plt.figure(figsize=(8.0, 6.0))
ax = fig.add_subplot(111, projection="3d")
surf_v = ax.plot_surface(
    kx, ky, valence_vals, cmap="Blues", linewidth=0, antialiased=True, alpha=0.9
)
surf_c = ax.plot_surface(
    kx, ky, conduction_vals, cmap="Reds", linewidth=0, antialiased=True, alpha=0.9
)
ax.set_xlabel(r"$k_x$ (1/$\AA$)")
ax.set_ylabel(r"$k_y$ (1/$\AA$)")
ax.set_zlabel("Energy (eV)")
ax.set_title("Valence (blue) and conduction (red) bands")
ax.view_init(elev=28, azim=-55)
plt.tight_layout()
plt.savefig("bands_3d.png", dpi=220)
plt.show()

