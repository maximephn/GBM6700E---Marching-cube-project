# Comparative Marching Cubes benchmark

Implementation of the proposal: Lorensen, Lewiner and 1 mm physical Gaussian
filtering plus Lewiner, at 0/20/50/100 HU of Gaussian noise.

```zsh
python -m pip install -r requirements.txt
python main.py single Task03_Liver/imagesTr/liver_0.nii.gz --roi-start-mm 80 110 90 --roi-size-mm 120 120 100 --visualize
python main.py benchmark Task03_Liver --patients 5 --seeds 20 --roi-start-mm 80 110 90 --roi-size-mm 120 120 100 --output results
```

Choose `--roi-start-mm` and `--roi-size-mm` to enclose the thoraco-lumbar
vertebrae. Results: observations (`measurements.csv`), aggregates (`summary.csv`),
mixed-effects/ Holm analysis (`mixed_effects_holm.csv`), and meshes (`.ply`) in
single mode. `--visualize` opens a direct interactive view for each algorithm:
raw 3D CT voxels (small cubes) beside the extracted 3D mesh. No HTML file is
saved in the output folder.
