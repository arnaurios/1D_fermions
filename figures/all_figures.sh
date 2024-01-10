#!/bin/bash
# CHOOSE DENSITY
figure_files=(plots_energy_panel.py  plots_components_energy_panel.py \ 
plots_density.py plots_density_panel.py plots_energy.py \
plots_occupations_panel.py  plots_occupations.py \
plots_3x1_energy_A2.py plots_correnergy_panel.py \
plots_denmat.py plots_denmat_panel.py plots_size_panel.py )

for file in ${figure_files[@]} 
do 
    
    echo $file
    python3 $file
done

python3 plots_2body_density_matrix_panel.py ML
python3 plots_denmat_panel.py ML

python3 plots_2body_density_matrix_panel.py HF
python3 plots_denmat_panel.py HF

python3 plots_2body_density_matrix_panel.py diag
python3 plots_denmat_panel.py diag

python3 plots_range.py +
python3 plots_range.py -
python3 plots_range_panel.py +
python3 plots_range_panel.py -
