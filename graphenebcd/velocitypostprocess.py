import argparse
import os
import time

import h5py
import numpy as np

# Physical Constants
EV_TO_HARTREE = 1.0 / 27.2113845
ANG_TO_BOHR = 1.8897259886
BOHR_TO_NM = 0.0529177210903

def unpack_matrix_fhi_aims(mom_k, num_bands):
    D_x = np.zeros((num_bands, num_bands), dtype=complex)
    D_y = np.zeros((num_bands, num_bands), dtype=complex)
    D_z = np.zeros((num_bands, num_bands), dtype=complex)

    linear_idx = 0
    # CRITICAL FIX: FHI-aims outputs Upper Triangular Row-Major
    for i in range(num_bands):
        for j in range(i, num_bands):
            D_x[i, j] = mom_k[linear_idx, 0] + 1j * mom_k[linear_idx, 1]
            D_y[i, j] = mom_k[linear_idx, 2] + 1j * mom_k[linear_idx, 3]
            D_z[i, j] = mom_k[linear_idx, 4] + 1j * mom_k[linear_idx, 5]

            if i == j:
                # Mathematically force diagonals to be purely imaginary 
                # to purge the residual DFT real-space grid noise.
                D_x[i, i] = 1j * np.imag(D_x[i, i])
                D_y[i, i] = 1j * np.imag(D_y[i, i])
                D_z[i, i] = 1j * np.imag(D_z[i, i])
            else:
                # The \nabla operator is anti-Hermitian
                D_x[j, i] = -np.conj(D_x[i, j])
                D_y[j, i] = -np.conj(D_y[i, j])
                D_z[j, i] = -np.conj(D_z[i, j])
            
            linear_idx += 1

    return D_x, D_y, D_z

def gaussian_delta(E, E_f, sigma=0.02):
    """Approximates the Fermi-Dirac derivative at zero temperature."""
    return (1.0 / (sigma * np.sqrt(np.pi))) * np.exp(-((E - E_f) / sigma)**2)

def parse_lattice_vectors_from_geometry(geometry_path):
    lattice = []
    with open(geometry_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0] == "lattice_vector":
                lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(lattice) == 3:
                    break
    if len(lattice) != 3:
        raise ValueError(f"Could not read 3 lattice vectors from {geometry_path}")
    return np.array(lattice, dtype=float)

def reciprocal_lattice_vectors(lattice_bohr):
    """Return reciprocal lattice vectors in bohr^-1 (with 2*pi convention)."""
    return 2.0 * np.pi * np.linalg.inv(lattice_bohr).T


def detect_bz_measure(recip_bohr_inv, k_points_dataset, tol=1e-12):
    """
    Detect whether sampling is effectively 2D or 3D from k_points and
    return the corresponding Brillouin-zone measure.
    """
    if k_points_dataset is None:
        # Conservative fallback: full 3D BZ volume.
        return "3D", 3, abs(np.linalg.det(recip_bohr_inv))

    # k_points stores (..., 4) with entries (i_k, kx, ky, kz) in fractional coords.
    k_frac = k_points_dataset[..., 1:4].reshape(-1, 3)
    spans = np.ptp(k_frac, axis=0)
    periodic_dirs = np.where(spans > tol)[0]

    if len(periodic_dirs) == 2:
        b1 = recip_bohr_inv[periodic_dirs[0]]
        b2 = recip_bohr_inv[periodic_dirs[1]]
        return "2D", 2, np.linalg.norm(np.cross(b1, b2))  # bohr^-2
    if len(periodic_dirs) == 3:
        return "3D", 3, abs(np.linalg.det(recip_bohr_inv))  # bohr^-3

    raise ValueError(
        "Could not infer periodic dimensionality from k_points dataset. "
        f"Detected {len(periodic_dirs)} periodic directions."
    )


def main():
    parser = argparse.ArgumentParser(description="Compute Berry curvature dipole from FHI-aims mommat.h5")
    parser.add_argument("--mommat", default="mommat.h5", help="Path to momentum-matrix HDF5 file")
    parser.add_argument("--geometry", default="geometry.in", help="Path to geometry.in for cell size")
    parser.add_argument("--sigma-eV", type=float, default=0.002, help="Gaussian width for -df/dE in eV")
    parser.add_argument("--eta-eV", type=float, default=0.002, help="Broadening eta in eV for Berry-curvature denominator")
    parser.add_argument("--fermi-shift-eV", type=float, default=0.0, help="Optional manual shift applied to Fermi energy")
    parser.add_argument("--progress-every", type=float, default=5.0, help="Print progress every N percent")
    parser.add_argument(
        "--output",
        default="bcdtensor_velocitypostprocess.txt",
        help="Output text file for computed BCD tensor (does not overwrite bcdtensor.txt by default)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mommat):
        raise FileNotFoundError(f"Cannot find momentum file: {args.mommat}")
    if not os.path.exists(args.geometry):
        raise FileNotFoundError(f"Cannot find geometry file: {args.geometry}")

    # 1. Load data
    with h5py.File(args.mommat, "r") as f:
        e_bands = f["E_bands"][:]  # (nkx, nky, nkz, num_bands, 1)
        mom_matrix = f["Momentummatrix"][:]  # (nkx, nky, nkz, n_pairs, 6)
        fermi_energy = float(f["Fermi_energy"][0]) + args.fermi_shift_eV
        k_points_dataset = f["k_points"][:] if "k_points" in f else None

    nkx, nky, nkz, num_bands, _ = e_bands.shape
    total_kpoints = nkx * nky * nkz
    expected_pairs = num_bands * (num_bands + 1) // 2
    if mom_matrix.shape[3] != expected_pairs:
        raise ValueError(
            f"Unexpected momentum shape: got {mom_matrix.shape[3]} pairs, expected {expected_pairs} for {num_bands} bands."
        )

    lattice_ang = parse_lattice_vectors_from_geometry(args.geometry)
    lattice_bohr = lattice_ang * ANG_TO_BOHR
    recip_bohr_inv = reciprocal_lattice_vectors(lattice_bohr)
    bz_dim, bz_dim_int, bz_measure = detect_bz_measure(recip_bohr_inv, k_points_dataset)
    if bz_measure <= 0.0:
        raise ValueError(f"Invalid Brillouin-zone {bz_dim} measure: {bz_measure}")

    bcd_tensor = np.zeros((3, 3))
    eta_Ha = args.eta_eV * EV_TO_HARTREE
    eta_sq = eta_Ha**2

    print(f"Processing {total_kpoints} k-points, {num_bands} bands.")
    print(f"Fermi Energy used: {fermi_energy:.6f} eV")
    if bz_dim == "2D":
        print(f"BZ area: {bz_measure:.6f} bohr^-2")
    else:
        print(f"BZ volume: {bz_measure:.6f} bohr^-3")
    print(f"Using PRL convention: integral measure d^{bz_dim_int}k/(2π)^{bz_dim_int}")
    progress_every = max(0.1, args.progress_every)
    next_progress = progress_every
    processed = 0
    t_start = time.time()

    # 2. Iterate over Brillouin-zone grid
    for kx in range(nkx):
        for ky in range(nky):
            for kz in range(nkz):
                energies_eV = e_bands[kx, ky, kz, :, 0]
                energies_Ha = energies_eV * EV_TO_HARTREE
                d_x, d_y, d_z = unpack_matrix_fhi_aims(mom_matrix[kx, ky, kz], num_bands)

                # 3. Kubo-like sum for Berry curvature and dipole
                for n in range(num_bands):
                    weight_eV = gaussian_delta(energies_eV[n], fermi_energy, sigma=args.sigma_eV)
                    weight_Ha = weight_eV * 27.2113845
                    if weight_Ha < 1e-6:
                        continue

                    o_x, o_y, o_z = 0.0, 0.0, 0.0
                    for m in range(num_bands):
                        if n == m:
                            continue
                        dE_sq = (energies_Ha[n] - energies_Ha[m]) ** 2
                        denom = dE_sq + eta_sq
                        o_x += 2.0 * np.imag(d_y[n, m] * d_z[m, n]) / denom
                        o_y += 2.0 * np.imag(d_z[n, m] * d_x[m, n]) / denom
                        o_z += 2.0 * np.imag(d_x[n, m] * d_y[m, n]) / denom

                    v_x = np.imag(d_x[n, n])
                    v_y = np.imag(d_y[n, n])
                    v_z = np.imag(d_z[n, n])
                    omega = np.array([o_x, o_y, o_z])
                    velocity = np.array([v_x, v_y, v_z])
                    bcd_tensor += np.outer(omega, velocity) * weight_Ha

                processed += 1
                progress_pct = 100.0 * processed / total_kpoints
                if progress_pct >= next_progress or processed == total_kpoints:
                    elapsed = time.time() - t_start
                    print(f"[progress] {progress_pct:6.2f}% ({processed}/{total_kpoints}) elapsed={elapsed:8.1f}s")
                    next_progress += progress_every

    # 4. Normalize discrete BZ integral with uniform k-point weight
    # PRL convention: ∫_k = ∫ d^d k / (2π)^d
    prefactor = bz_measure / ((2.0 * np.pi) ** bz_dim_int * total_kpoints)
    bcd_tensor *= prefactor

    print("-" * 50)
    print("NON-LINEAR TOPOLOGICAL INTEGRATION COMPLETE")
    print("-" * 50)
    print("Berry Curvature Dipole Tensor D_ab (atomic units):")
    print(bcd_tensor)
    nm_tensor = None
    if bz_dim == "2D":
        nm_tensor = bcd_tensor * BOHR_TO_NM
        print("Same tensor converted to nm (2D length convention):")
        print(nm_tensor)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("Berry Curvature Dipole postprocessing result\n")
        f.write(f"mommat_file: {args.mommat}\n")
        f.write(f"geometry_file: {args.geometry}\n")
        f.write(f"dimension_detected: {bz_dim}\n")
        f.write(f"num_kpoints: {total_kpoints}\n")
        f.write(f"num_bands: {num_bands}\n")
        f.write(f"fermi_energy_eV_used: {fermi_energy:.10f}\n")
        f.write(f"sigma_eV: {args.sigma_eV}\n")
        f.write(f"eta_eV: {args.eta_eV}\n")
        f.write(f"bz_measure: {bz_measure:.12e}\n")
        f.write(f"prl_prefactor: {prefactor:.12e}\n")
        f.write("tensor_atomic_units:\n")
        f.write(np.array2string(bcd_tensor, precision=12, suppress_small=False))
        f.write("\n")
        if nm_tensor is not None:
            f.write("tensor_nm:\n")
            f.write(np.array2string(nm_tensor, precision=12, suppress_small=False))
            f.write("\n")
    print(f"Saved tensor output to: {args.output}")


if __name__ == "__main__":
    main()
