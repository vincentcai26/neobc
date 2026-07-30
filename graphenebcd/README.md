# Uniaxially Strained Graphene Berry Curvature Dipole Calculation

These are scripts to run with FHI-aims to compute Berry Curvature Dipole (BCD) of uniaxially strained graphene with a Semenoff Mass. 

In order for graphene to have a BCD, it must have BOTH uniaxial strain, and a Semenoff mass. 

There are two ways to implement this:

## 1. From FHI-aims, `compute_momentummatrix`

This will directly compute the momentum matrices. Notice that SOC is not supported here, and you need HDF5 compatibility. This is a 3-step FHI-aims process: 

1. Relax
2. SCF
3. NSCF (from the momentum matrices)

All three input files can be generated in `generateinputs.py`. To run the full sequence: 

1. Relax

Set the mode to "relax", set the strain and the Semenoff mass offset. This literally changes the charge of the Nuclei on one sublattice vs the other, to represent a Semenoff mass, which is simply a potential energy difference on sites of one sublattice vs. the other. Choose a very coarse k-grid, like (18,18).

Run the script `generateinputs.py` with this mode and k-grid (the only two parameters, just edit the function call at the end of the script). Will generate correct control.in and geometry.in. Then run FHI-aims in this directory. 

Outputs the correct geometry to geometry.in.next_step, like all FHI-aims relax calculations. 

2. SCF

Set the mode to "scf", with same strain and Semenoff mass offset. Set k-grid to something slightly denser like (42,42). 

Run the script `generateinputs.py`, which will generate the control.in, and literally just copy the geometry.in.next_step into geometry.in. Then run FHI-aims. 

Outputs both the 32 `aims_restart_XXX` (between 000 and 031) files, that are the SCF results, input into the NSCF (next step), as well as four other files that are the real space H(R) and S(R) (Hamiltonian and Overlap) matrices data. Is not useful for this, but for the other method. 

Will also output a `ri_restart_coeffs`. This is important, don't touch this. This is a file that the resolution of identity (RI) algorithm behind the scenes will use to construct SCF results for the different k-grid you define for NSCF calculations. 


3. NSCF

Set mode to "nscf_momentum", with same strain and Semenoff mass offset (strain doesn't really matter since you have the geometry, but you NEED the same Semeneoff mass offset). Set k-grid to somethign dense like (400,400). 

You need HDF5 compatibility only for this step, since it will output the `mommat.h5` file. Thus, you need to move your whole repo to the `/dev/shm/` directory on your machine so that FHI-aims does not stall. See notes on how to do this. 

Then run FHI-aims in that repo, and copy over `mommat.h5` back into this directory (can also copy logs like `aims.out` or whatever. I think the hessian is also printed here, but not important). 

4. Post process

Simply run `velocitypostprocess.py` and it will spit out the $3 \times 3$ BCD tensor. 

May need to adjust sigma (width of delta function, for zero temperature Fermi-Dirac derivative, don't set too small or large) and eta (denominator regularization in Kubo formula, with the E_n-E_m, if it is zero) in the script 

## 2. Custom Python Script from H(R) and S(R)


For this, run up to step 2 of above (the relax and SCF), and then run two more scripts: 

1. `parse_rs_play.py`: parses the plain text files for the H(R) and S(R) matrices. Outputs a file named `parsed_rs_matrices.npz`. Just directly run this, will look for files outputted from the output_rs_matrices directive. 

2. `postprocess.py`: constructs tight-binding (orthogonalized H(k)) matrices to compute band structure. Still a work in progress for Berry curvature and fermi velocity. Up to now will generate image files for band structure plots. 

Note that this whole method relies on the fact that the effective ``dipole matrices", or <u_nk|r|u_mk> are neglible, which is generally not true for materials with strong orbital hybridization. So this method is strictly less accurate than the last. Only advantage is perhaps slightly more scalable k-grid integration for Berry curvature across the BZ, but in practice this script is actually pretty slow, and NSCF is pretty fast up to about 400 x 400 k-grid in 2D. 