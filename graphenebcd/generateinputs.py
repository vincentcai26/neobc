import numpy as np
import os
import re


def generate_fhi_aims_inputs(
    strain_rate,
    kgrid=(18, 18),
    poisson_ratio=0.165,
    a0=2.46,
    basis_tier="intermediate",
    mode="relax",
    isRelax=None,
    next_step_geometry="geometry.in.next_step",
    momentum_energy_window=(-10.0, 10.0),
    momentum_kpoint=0,
    semenoff_delta=0.0,
):
    """
    Generate geometry.in and control.in for uniaxially strained graphene.

    Parameters:
    - strain_rate: uniaxial strain value (e.g. 0.03 for 3%).
    - kgrid: in-plane Monkhorst-Pack grid as (kx, ky).
    - poisson_ratio: Poisson ratio for graphene.
    - a0: pristine lattice constant in Angstrom.
    - basis_tier: FHI-aims species-default tier (e.g. light/intermediate/tight).
    - mode: one of {"relax", "scf", "nscf_momentum"}.
      * relax         -> writes strained primitive geometry + relax_geometry.
      * scf           -> copies next_step_geometry + static SCF + writes RI-density restart.
      * nscf_momentum -> reads RI-density restart and runs non-self-consistent momentum output.
    - isRelax: legacy boolean mode selector (optional). If set, overrides mode:
      * True  -> relax
      * False -> scf
    - next_step_geometry: source geometry file used for "scf" / "nscf_momentum".
    - momentum_energy_window: (Emin, Emax) for compute_momentummatrix in eV.
    - momentum_kpoint: k-point index for compute_momentummatrix (0 = all k-points to HDF5).
    - semenoff_delta: sublattice charge shift for Semenoff mass.
      If > 0, writes two species C_A/C_B with:
      * nucleus(C_A/C_B) = 6 +/- semenoff_delta
      * valence 2p and ion_occ 2p = 2 +/- semenoff_delta
    """
    if isRelax is not None:
        if not isinstance(isRelax, bool):
            raise TypeError("isRelax must be a boolean when provided")
        mode = "relax" if isRelax else "scf"

    valid_modes = {"relax", "scf", "nscf_momentum"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")
    if semenoff_delta < 0.0:
        raise ValueError("semenoff_delta must be >= 0")

    # ==========================================
    # 1. Strain and Geometry Calculations
    # ==========================================
    # Pristine Cartesian lattice vectors (zigzag along x-axis)
    a1_pristine = np.array([a0 * np.sqrt(3) / 2, -a0 * 1 / 2, 0.0])
    a2_pristine = np.array([a0 * np.sqrt(3) / 2, a0 * 1 / 2, 0.0])

    # Fractional coordinates for the two sublattices
    fA = np.array([0.0, 0.0, 0.0])
    fB = np.array([1 / 3, 1 / 3, 0.0])

    tauA_pristine = fA[0] * a1_pristine + fA[1] * a2_pristine
    tauB_pristine = fB[0] * a1_pristine + fB[1] * a2_pristine

    # Uniaxial strain along x with Poisson contraction along y
    strain_matrix = np.array(
        [
            [1.0 + strain_rate, 0.0, 0.0],
            [0.0, 1.0 - (poisson_ratio * strain_rate), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    a1_strained = strain_matrix @ a1_pristine
    a2_strained = strain_matrix @ a2_pristine
    a3_strained = np.array([0.0, 0.0, 15.0])  # vacuum padding

    tauA_strained = strain_matrix @ tauA_pristine
    tauB_strained = strain_matrix @ tauB_pristine

    # ==========================================
    # 2. Parse Species Defaults
    # ==========================================
    species_file = (
        f"/home/ubuntu/software/fhiaims/species_defaults/defaults_2020/{basis_tier}/06_C_default"
    )
    if not os.path.exists(species_file):
        raise FileNotFoundError(f"Cannot locate FHI-aims species defaults at: {species_file}")

    with open(species_file, "r", encoding="utf-8") as f:
        c_default_text = f.read()

    def _build_species_text(name: str, z_eff: float, p_occ: float) -> str:
        num_pattern = r"[+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?"
        text = re.sub(
            r"^\s*species\s+\S+\b.*$",
            f"  species        {name}",
            c_default_text,
            count=1,
            flags=re.MULTILINE,
        )
        text = re.sub(
            rf"^(\s*nucleus\s+){num_pattern}\s*$",
            rf"\g<1>{z_eff:.8f}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        text = re.sub(
            rf"^(\s*valence\s+2\s+p\s+){num_pattern}\s*$",
            rf"\g<1>{p_occ:.8f}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        return text

    use_split_species = semenoff_delta > 0.0
    if use_split_species:
        c_a_text = _build_species_text("C_A", 6.0 + semenoff_delta, 2.0 + semenoff_delta)
        c_b_text = _build_species_text("C_B", 6.0 - semenoff_delta, 2.0 - semenoff_delta)
    else:
        c_text = _build_species_text("C", 6.0, 2.0)

    # ==========================================
    # 3. Write geometry.in
    # ==========================================
    if mode == "relax":
        with open("geometry.in", "w", encoding="utf-8") as f:
            f.write("# Strained Graphene (zigzag-axis strain), primitive cell\n")
            f.write(f"lattice_vector {a1_strained[0]:.8f} {a1_strained[1]:.8f} {a1_strained[2]:.8f}\n")
            f.write(f"lattice_vector {a2_strained[0]:.8f} {a2_strained[1]:.8f} {a2_strained[2]:.8f}\n")
            f.write(f"lattice_vector {a3_strained[0]:.8f} {a3_strained[1]:.8f} {a3_strained[2]:.8f}\n")
            if use_split_species:
                f.write(f"atom {tauA_strained[0]:.8f} {tauA_strained[1]:.8f} {tauA_strained[2]:.8f} C_A\n")
                f.write(f"atom {tauB_strained[0]:.8f} {tauB_strained[1]:.8f} {tauB_strained[2]:.8f} C_B\n")
            else:
                f.write(f"atom {tauA_strained[0]:.8f} {tauA_strained[1]:.8f} {tauA_strained[2]:.8f} C\n")
                f.write(f"atom {tauB_strained[0]:.8f} {tauB_strained[1]:.8f} {tauB_strained[2]:.8f} C\n")
    else:
        if not os.path.exists(next_step_geometry):
            raise FileNotFoundError(f"Cannot find next-step geometry file: {next_step_geometry}")
        with open(next_step_geometry, "r", encoding="utf-8") as src, open(
            "geometry.in", "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        if use_split_species:
            with open("geometry.in", "r", encoding="utf-8") as f:
                geom_text = f.read()
            if " C_A" not in geom_text or " C_B" not in geom_text:
                raise ValueError(
                    "semenoff_delta > 0 requires geometry with C_A/C_B species labels. "
                    "Run mode='relax' first with same semenoff_delta, then use geometry.in.next_step."
                )

    # ==========================================
    # 4. Write control.in
    # ==========================================
    with open("control.in", "w", encoding="utf-8") as f:
        f.write("# FHI-aims control parameters for strained graphene\n")
        f.write("xc pbe\n")
        f.write("spin none\n")
        f.write("relativistic atomic_zora scalar\n")
        f.write(f"k_grid {kgrid[0]} {kgrid[1]} 1\n")
        f.write("empty_states 15\n")

        if mode == "relax":
            f.write("relax_geometry bfgs 1e-3\n")
        else:
            if mode == "scf":
                f.write("output_rs_matrices plain\n")
                f.write("ri_density_restart write\n")
            if mode == "nscf_momentum":
                f.write("sc_iter_limit 0\n")
                f.write("ri_density_restart read 0\n")
                f.write(
                    "compute_momentummatrix "
                    f"{momentum_energy_window[0]} {momentum_energy_window[1]} {momentum_kpoint}\n"
                )

        f.write("\n")
        f.write("################################################################################\n")
        f.write(f"# Species Defaults: Carbon ({basis_tier})\n")
        f.write("################################################################################\n")
        if use_split_species:
            f.write(c_a_text + "\n")
            f.write("\n")
            f.write(c_b_text + "\n")
        else:
            f.write(c_text + "\n")

    print(f"Successfully generated FHI-aims inputs for strain = {strain_rate * 100:.2f}%.")
    if mode == "relax":
        print("Mode: relax (strained geometry + relax_geometry)")
    elif mode == "scf":
        print(
            "Mode: scf "
            f"(geometry copied from {next_step_geometry} + output_rs_matrices plain + ri_density_restart write)"
        )
    else:
        print(
            "Mode: nscf_momentum "
            "(geometry copied from "
            f"{next_step_geometry} + sc_iter_limit 0 + ri_density_restart read 0 + compute_momentummatrix)"
        )
    print(f"Basis set tier utilized: {basis_tier}")


if __name__ == "__main__":
    generate_fhi_aims_inputs(strain_rate=0.05, semenoff_delta=0.05, kgrid=(400,400), mode="nscf_momentum")