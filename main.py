#!/usr/bin/env python3
"""Benchmark Marching Cubes : Lorensen, Lewiner et Lewiner après filtre gaussien.

Exemples:
  python main.py single Task03_Liver/imagesTr/liver_0.nii.gz --roi-start-mm 80 110 90 --roi-size-mm 120 120 100
  python main.py benchmark Task03_Liver --patients 5 --seeds 20 --roi-start-mm 80 110 90 --roi-size-mm 120 120 100
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from plotly.subplots import make_subplots

ISO_LEVEL_HU = 250.0
NOISE_LEVELS = (0.0, 20.0, 50.0, 100.0)
PIPELINES = ("lorensen", "lewiner", "gaussian_lewiner")

@dataclass
class MeshMetrics:
    vertices: int; faces: int; edges: int; components: int
    euler_characteristic: int; euler_per_component: float
    boundary_edges: int; watertight: bool; volume_mm3: float


def vec(values, name, allow_zero=False):
    """
    Returns and checks that the values provided are a valid position vector and returns these values as a 3-dimensional numpy array
    """
    if values is None:
        return None
    if len(values) != 3 or any(v < 0 for v in values) or (not allow_zero and any(v == 0 for v in values)):
        raise ValueError(f"{name} doit contenir trois valeurs {'non négatives' if allow_zero else 'positives'}.")
    return np.asarray(values, dtype=float)


def crop(volume, spacing, start_mm, size_mm):
    """
    Crops the physical ROI (region of interest). Without ROI, central crop of 120 mm (backup solution)
    """
    if (start_mm is None) != (size_mm is None):
        raise ValueError("--roi-start-mm et --roi-size-mm doivent être fournis ensemble.")
    shape = np.asarray(volume.shape[:3])
    if start_mm is None:
        size_mm = np.minimum(np.full(3, 120.), shape * spacing)
        start_mm = np.maximum((shape * spacing - size_mm) / 2., 0.)
    start = np.clip(np.floor(start_mm / spacing).astype(int), 0, shape - 2)
    stop = np.clip(np.ceil((start_mm + size_mm) / spacing).astype(int), start + 2, shape)
    return np.asarray(volume[tuple(slice(a, b) for a, b in zip(start, stop))], dtype=np.float32), start * spacing


def load_roi(path, start_mm, size_mm):
    """
    Loads ROI by croping the volume in the region of interest, 
    and returns the croped volume, the spacing and the offset
    """
    image = nib.load(str(path)); volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"{path} n'est pas un volume 3D.")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    roi, offset = crop(volume, spacing, start_mm, size_mm)
    return roi, spacing, offset

def noise(volume, sigma_hu, seed):
    """
    Generates a noisy volume (gaussian noise)
    """
    return volume + np.random.default_rng(seed).normal(0, sigma_hu, volume.shape).astype(np.float32)


def physical_gaussian(volume, spacing, sigma_mm=1.0):
    """
    Sigma isotrope en mm, converti en sigma par axe voxel
    """
    return gaussian_filter(volume, sigma=sigma_mm / spacing, mode="nearest")


def merge_vertices(vertices, faces, tolerance_mm=1e-4):
    """
    Nettoyage explicite: fusion de sommets, suppression faces dégénérées/dupliquées
    """
    
    key = np.rint(vertices / tolerance_mm).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    merged = np.column_stack([np.bincount(inverse, weights=vertices[:, i]) / count for i in range(3)])
    faces = inverse[faces]
    faces = faces[np.array([len(set(face)) == 3 for face in faces])]
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return merged, faces[np.sort(keep)]


def mesh_from_volume(volume, spacing, offset, method):
    vertices, faces, _, _ = marching_cubes(volume, ISO_LEVEL_HU, spacing=tuple(spacing), method=method, allow_degenerate=False) # uses scikit 
    vertices, faces = merge_vertices(vertices + offset, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)


def boundary_edges(mesh):
    if not len(mesh.faces): return 0
    _, counts = np.unique(np.sort(mesh.edges_sorted, axis=1), axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def metrics(mesh):
    parts = list(mesh.split(only_watertight=False)) if len(mesh.faces) else []
    v, e, f = len(mesh.vertices), len(mesh.edges_unique), len(mesh.faces)
    chi = v - e + f
    return MeshMetrics(v, f, e, len(parts), int(chi), chi / len(parts) if parts else np.nan,
                       boundary_edges(mesh), bool(mesh.is_watertight), abs(float(mesh.volume)) if mesh.is_watertight else np.nan)


def largest(mesh):
    parts = list(mesh.split(only_watertight=False))
    return max(parts, key=lambda item: len(item.faces)) if parts else mesh


def distances(candidate, reference):
    """HD95 bidirectionnel et distance de surface moyenne, en mm."""
    a, b = largest(candidate).vertices, largest(reference).vertices
    if not len(a) or not len(b): return np.nan, np.nan
    d = np.r_[cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]]
    return float(np.percentile(d, 95)), float(np.mean(d))


def pipeline(volume, spacing, offset, name):
    started = time.perf_counter()
    if name == "gaussian_lewiner":
        volume, method = physical_gaussian(volume, spacing), "lewiner"
    else: method = name
    return mesh_from_volume(volume, spacing, offset, method), time.perf_counter() - started


def voxel_blocks(volume, spacing, offset, maximum=5000):
    """Construit un maillage de petits cubes pour visualiser les voxels bruts.

    Le plafonnement garde l'affichage interactif; il n'a aucune incidence sur
    l'algorithme Marching Cubes, qui utilise toujours tous les voxels de la ROI.
    """
    indices = np.argwhere(volume >= ISO_LEVEL_HU)
    if not len(indices):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0)
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]
    centers = offset + indices * spacing
    half = spacing * 0.43
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    vertices = (centers[:, None, :] + corners[None, :, :] * half).reshape(-1, 3)
    cube_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                           [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                           [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    faces = (cube_faces[None, :, :] + 8 * np.arange(len(indices))[:, None, None]).reshape(-1, 3)
    intensity = np.repeat(volume[tuple(indices.T)], 8)
    return vertices, faces, intensity


def visualize_input_vs_output(volume, spacing, offset, mesh, pipeline_name):
    """Ouvre la comparaison directe entre voxels 3D et surface extraite."""
    voxel_vertices, voxel_faces, intensity = voxel_blocks(volume, spacing, offset)
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Entrée — voxels CT bruts (≥ 250 HU)", f"Sortie — maillage {pipeline_name}"),
    )
    figure.add_trace(go.Mesh3d(x=voxel_vertices[:, 0], y=voxel_vertices[:, 1], z=voxel_vertices[:, 2],
                               i=voxel_faces[:, 0], j=voxel_faces[:, 1], k=voxel_faces[:, 2],
                               intensity=intensity, colorscale="Gray", opacity=0.72,
                               colorbar_title="HU", name="Voxels bruts"), row=1, col=1)
    figure.add_trace(go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                               i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                               color="lightsteelblue", opacity=0.85, name=pipeline_name), row=1, col=2)
    figure.update_scenes(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)", aspectmode="data", row=1, col=1)
    figure.update_scenes(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)", aspectmode="data", row=1, col=2)
    figure.update_layout(title=f"Marching Cubes : volume voxelisé → surface ({pipeline_name})", height=700)
    figure.show()


def patient_paths(dataset, number):
    with (dataset / "dataset.json").open() as handle: entries = json.load(handle)["training"]
    paths = sorted(dataset / item["image"].removeprefix("./") for item in entries)
    if not 1 <= number <= len(paths): raise ValueError(f"--patients doit être entre 1 et {len(paths)}.")
    return paths[:number]


def holm(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p); adjusted = np.empty_like(p); running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index]); adjusted[index] = min(running, 1.)
    return adjusted


def mixed_effects(frame, output):
    try: import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels absent: analyse LME ignorée."); return
    rows = []
    for metric_name in ("hd95_mm", "mean_surface_distance_mm", "runtime_s", "volume_mm3", "euler_per_component", "boundary_edges"):
        data = frame.dropna(subset=[metric_name, "patient"])
        if len(data) < 10 or data.patient.nunique() < 2: continue
        try:
            fit = smf.mixedlm(f"{metric_name} ~ C(pipeline) * C(noise_sigma_hu)", data, groups=data.patient).fit(reml=False, method="lbfgs", disp=False)
            rows.extend({"metric": metric_name, "term": term, "coefficient": value, "p_value": pvalue}
                        for term, value, pvalue in zip(fit.params.index, fit.params, fit.pvalues) if term != "Group Var")
        except (ValueError, np.linalg.LinAlgError) as error: print(f"LME {metric_name} impossible: {error}")
    if rows:
        table = pd.DataFrame(rows); table["p_holm"] = holm(table.p_value); table.to_csv(output / "mixed_effects_holm.csv", index=False)


def benchmark(args):
    start_mm, size_mm = vec(args.roi_start_mm, "origine ROI", True), vec(args.roi_size_mm, "taille ROI")
    rows = []
    for path in patient_paths(args.dataset, args.patients):
        patient = path.stem.replace(".nii", ""); print(f"Patient {patient}")
        clean, spacing, offset = load_roi(path, start_mm, size_mm)
        reference, _ = pipeline(clean, spacing, offset, "lewiner")
        for sigma in NOISE_LEVELS:
            for seed in range(args.seeds):
                noisy = noise(clean, sigma, seed)
                for name in PIPELINES:
                    try:
                        mesh, runtime = pipeline(noisy, spacing, offset, name); hd95, mean = distances(mesh, reference)
                        rows.append({"patient": patient, "seed": seed, "noise_sigma_hu": sigma, "pipeline": name,
                                     "runtime_s": runtime, "hd95_mm": hd95, "mean_surface_distance_mm": mean, **asdict(metrics(mesh))})
                    except (RuntimeError, ValueError) as error:
                        rows.append({"patient": patient, "seed": seed, "noise_sigma_hu": sigma, "pipeline": name, "error": str(error)})
    frame = pd.DataFrame(rows); args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "measurements.csv", index=False)
    frame.groupby(["pipeline", "noise_sigma_hu"]).mean(numeric_only=True).to_csv(args.output / "summary.csv")
    mixed_effects(frame, args.output)


def single(args):
    start_mm, size_mm = vec(args.roi_start_mm, "origine ROI", True), vec(args.roi_size_mm, "taille ROI")
    clean, spacing, offset = load_roi(args.image, start_mm, size_mm); noisy = noise(clean, args.noise_sigma, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in PIPELINES:
        try: mesh, runtime = pipeline(noisy, spacing, offset, name)
        except ValueError as error:
            print(f"{name}: aucune isosurface à {ISO_LEVEL_HU:g} HU dans cette ROI ({error})"); continue
        mesh.export(args.output / f"{args.image.stem}_{name}.ply")
        if args.visualize:
            visualize_input_vs_output(noisy, spacing, offset, mesh, name)
        print(f"{name}: {len(mesh.vertices)} sommets, {len(mesh.faces)} faces, {runtime:.2f} s")

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--roi-start-mm", type=float, nargs=3, metavar=("X", "Y", "Z")); common.add_argument("--roi-size-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    common.add_argument("--output", type=Path, default=Path("results"))
    one = sub.add_parser("single", parents=[common]); one.add_argument("image", type=Path); one.add_argument("--noise-sigma", type=float, default=50.); one.add_argument("--seed", type=int, default=0); one.add_argument("--visualize", action="store_true", help="Ouvre les vues voxels 3D vs maillage, sans sauvegarder de HTML."); one.set_defaults(func=single)
    many = sub.add_parser("benchmark", parents=[common]); many.add_argument("dataset", type=Path); many.add_argument("--patients", type=int, default=5); many.add_argument("--seeds", type=int, default=20); many.set_defaults(func=benchmark)
    return parser

if __name__ == "__main__":
    arguments = cli().parse_args(); arguments.func(arguments)
