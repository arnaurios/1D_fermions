# 1D_fermions
Repository for codes and data for arXiv:2304.04725. The data is structured in folders containing data, figures and codes, as described below.

## Data folders
+ ``neural_network``: the folders in the directory ``neural_network`` contain the data for energies obtained with the Neural Quantum State described in the paper.  
+ ``Diagonalization``: the folders within the ``Diagonalization`` directory contain the data obtained with the direct diagonalization approach.

## Code folders
### A2_perturbation_theory
The python code and Jupyter notebook [A2_particles_HO_Gaussian_PT.ipynb](./A2_perturbation_theory/A2_particles_HO_Gaussian_PT.ipynb)
 in this folder solve the 2-body problem of spinless fermions with a gaussian interaction by employing **perturbation theory** on the interaction.

### A2_space_solution
The python code and Jupyter notebook [A2_space_solution.ipynb](./A2_space_solution/A2_space_solution.ipynb)
in this folder solves the 2-body problem of spinless fermions with a gaussian interaction by employing a discetization approach on real space.

### hartree_fock
The python code in this folder solves the many-body problem of spinless fermions with a gaussian interaction by employing the Hartree-Fock method. The same discetization approach on real space is used as in the previous folder, but the many-body results are available. 


## Figure folder
The folders in the directory ``figures`` generates the figures in the paper in pdf format. The .py files are matplotlib python scripts which work with Python3 (eg ``python3 plots_energy.py``. Some of these files require input arguments. To generate all the figures, we provide a bash script ``all_figures.sh``.



