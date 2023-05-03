# 1D_fermions
Repository for codes and data for arXiv:2304.04725. The data is structured in folders containing data, figures and codes, as described below.

## Data folders
The folders in the directory ``neural_network`` contain the data for energies obtained with the Neural Quantum State described in the paper. 
The folders within the ``Diagonalization`` directory contain the data obtained with the direct diagonalization approach.

## Code folders
### A2_perturbation_theory

The python code and Jupyter notebook [A2_particles_HO_Gaussian_PT.ipynb](./A2_perturbation_theory/A2_particles_HO_Gaussian_PT.ipynb)
 in this folder solve the 2-body problem of spinless fermions with a gaussian interaction by employing **perturbation theory** on the interaction.



## Figure folder
The folders in the directory ``figures`` generates the figures in the paper in pdf format. The .py files are matplotlib python scripts which work with Python3 (eg ``python3 plots_energy.py``. Some of these files require input arguments. To generate all the figures, we provide a bash script ``all_figures.sh``.



