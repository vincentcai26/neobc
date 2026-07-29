import numpy as np
import os
import re


def generate_fhi_aims_inputs(
    strain_rate,
    kgrid=(18, 18),
    poisson_ratio=0.165,
    a0=2.46,
    basis_tier="intermediate",
    isRelax=True,
    next_step_geometry="geometry.in.next_step",
):
    """
    Generate geometry.in and control.in for uniaxially strained graphene.

    Parameters:
    - strain_rate: uniaxial strain value (e.g. 0.03 for 3%).
    - kgrid: in-plane Monkhorst-Pack grid as (kx, ky).
    - poisson_ratio: Poisson ratio for graphene.
    - a0: pristine lattice constant in Angstrom.
    - basis_tier: FHI-aims species-default tier (e.g. light/intermediate/tight).
    - isRelax: boolean mode selector.
      * True  -> writes strained primitive geometry and relax_geometry keyword.
      * False -> copies next_step_geometry and writes output_rs_matrices plain.
    - next_step_geometry: source geometry file used when isRelax=False.
    """
    if not isinstance(isRelax, bool):
        raise TypeError("isRelax must be a boolean")

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

    # Keep single neutral carbon species only.
    c_text = re.sub(
        r"^\s*species\s+C\b.*$",
        "  species        C",
        c_default_text,
        count=1,
        flags=re.MULTILINE,
    )

    # ==========================================
    # 3. Write geometry.in
    # ==========================================
    if isRelax:
        with open("geometry.in", "w", encoding="utf-8") as f:
            f.write("# Strained Graphene (zigzag-axis strain), primitive cell\n")
            f.write(f"lattice_vector {a1_strained[0]:.8f} {a1_strained[1]:.8f} {a1_strained[2]:.8f}\n")
            f.write(f"lattice_vector {a2_strained[0]:.8f} {a2_strained[1]:.8f} {a2_strained[2]:.8f}\n")
            f.write(f"lattice_vector {a3_strained[0]:.8f} {a3_strained[1]:.8f} {a3_strained[2]:.8f}\n")
            f.write(f"atom {tauA_strained[0]:.8f} {tauA_strained[1]:.8f} {tauA_strained[2]:.8f} C\n")
            f.write(f"atom {tauB_strained[0]:.8f} {tauB_strained[1]:.8f} {tauB_strained[2]:.8f} C\n")
    else:
        if not os.path.exists(next_step_geometry):
            raise FileNotFoundError(f"Cannot find next-step geometry file: {next_step_geometry}")
        with open(next_step_geometry, "r", encoding="utf-8") as src, open(
            "geometry.in", "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

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

        if isRelax:
            f.write("relax_geometry bfgs 1e-3\n")
        else:
            f.write("output_rs_matrices plain\n")

        f.write("\n")
        f.write("################################################################################\n")
        f.write(f"# Species Defaults: Carbon ({basis_tier})\n")
        f.write("################################################################################\n")
        f.write(c_text + "\n")

    print(f"Successfully generated FHI-aims inputs for strain = {strain_rate * 100:.2f}%.")
    if isRelax:
        print("Mode: relax (strained geometry + relax_geometry)")
    else:
        print(f"Mode: scf (geometry copied from {next_step_geometry} + output_rs_matrices plain)")
    print(f"Basis set tier utilized: {basis_tier}")


if __name__ == "__main__":
    generate_fhi_aims_inputs(strain_rate=0.05, kgrid=(200, 200), isRelax=False)