import os
import shutil
import numpy as np
import pyvista as pv
import vtk
from shapely.geometry import Polygon
from shapely import vectorized
from stl import mesh


# curvature-based LE / TE finder
def _compute_curvature(poly, kind="mean"):
    curv = vtk.vtkCurvatures()
    curv.SetInputData(poly)
    if kind == "mean":
        curv.SetCurvatureTypeToMean()
    else:
        curv.SetCurvatureTypeToGaussian()
    curv.Update()
    return pv.wrap(curv.GetOutput())


def _find_le_te(poly, top_k=20):
    curved = _compute_curvature(poly, "mean")
    k  = curved.point_data["Mean_Curvature"]
    p  = curved.points
    idx = np.argsort(np.abs(k))[-top_k:]
    top = p[idx]

    i, j = np.unravel_index(
        np.argmax(np.linalg.norm(top[:, None] - top[None], axis=2)),
        (top_k, top_k)
    )
    le, te = top[i], top[j]
    if le[0] > te[0]:
        le, te = te, le
    return le, te, curved


# split STL → {slat, main, flap}   (assumes 3 bodies for now)
def _label_by_centroid(parts):
    order  = np.argsort([p.center[0] for p in parts])
    return dict(zip(("slat", "main", "flap"), [parts[i] for i in order]))

 
def detect_geometry_info(stl_path):
    """Return dict[name] → {LE_point, TE_point, mesh, chord_vector}."""
    parts = (
        pv.read(stl_path)
        .connectivity(split_bodies=True)
        .split_bodies()
    )
    if len(parts) != 3:
        raise RuntimeError("expecting exactly 3 solid parts")

    info = {}
    for name, body in _label_by_centroid(parts).items():
        surf          = body.extract_surface().triangulate()
        le, te, mesh  = _find_le_te(surf)
        info[name] = {
            "LE_point": le,
            "TE_point": te,
            "mesh":     mesh,
            "chord_vector": (te - le) / np.linalg.norm(te - le),
        }
    return info


# true 2-D boundary loop (for gap buffer)
def boundary_loop_2d(poly):
    """Return ordered Nx2 array of the outer boundary of poly."""
    edges = poly.extract_feature_edges(boundary_edges=True)
    pts2d = edges.points[:, :2]
    conn  = edges.lines.reshape(-1, 3)[:, 1:]
    # build adjacency & trace simple loop
    nxt   = {a: b for a, b in conn}
    start = conn[0, 0]
    loop  = [start]
    while True:
        nxt_idx = nxt.get(loop[-1])
        if nxt_idx is None or nxt_idx == start:
            break
        loop.append(nxt_idx)
    loop.append(start)
    return pts2d[loop]


# polygon-buffer gap check  (new, all bodies as polygons)

def scale_to_unit(src, dst):
    g  = detect_geometry_info(src); 
    le, te = g["main"]["LE_point"], g["main"]["TE_point"]
    s = 1.0/np.linalg.norm(te - le)
    m = mesh.Mesh.from_file(src)
    m.vectors[:] = ((m.vectors.reshape(-1,3)-le)*s).reshape(-1,3,3)
    os.makedirs(os.path.dirname(dst), exist_ok=True); m.save(dst)
    print(f"scaled chord to 1 m, scale {s:.4f})")

def stretch_transform(input_stl, output_stl, new_thickness=0.5):
    # Load the STL mesh
    m = mesh.Mesh.from_file(input_stl)
    # Reshape to (num_vertices, 3)
    vertices = m.vectors.reshape(-1, 3)
    
    # Compute current z-range
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    current_thickness = z_max - z_min
    # print(f"Before stretching: z_min = {z_min:.5f}, z_max = {z_max:.5f}, thickness = {current_thickness:.5f}")
    
    if current_thickness == 0:
        raise ValueError("Mesh appears flat in the z-direction, cannot stretch.")
    
    factor = new_thickness / current_thickness
    # print(f"Scaling factor: {factor:.5f}")
    
    # First scale z values to get the desired thickness, then center them so that z=0 is the midplane
    vertices[:, 2] = (vertices[:, 2] - z_min) * factor - new_thickness/2
    m.vectors = vertices.reshape(-1, 3, 3)
    
    # Verify new thickness
    new_z_min = vertices[:, 2].min()
    new_z_max = vertices[:, 2].max()
    new_thick = new_z_max - new_z_min
    # print(f"After stretching and centering: z_min = {new_z_min:.5f}, z_max = {new_z_max:.5f}, thickness = {new_thick:.5f}")
    
    # Save the transformed STL
    m.save(output_stl)

def split_stl_case(input_stl, case_folder, case_number):
    """
    Read input_stl, split it into its three parts (assumed to be slat, main, flap),
    and write the following files in case_folder:
      - combined{case_number}.stl (a copy of the input_stl)
      - slat{case_number}.stl
      - main{case_number}.stl
      - flap{case_number}.stl
    """
    os.makedirs(case_folder, exist_ok=True)
    # Copy the input file as the combined airfoil.
    combined_file = os.path.join(case_folder, f"combined{case_number}.stl")
    shutil.copy(input_stl, combined_file)
    # print(f"[ok] saved STLs to {combined_file}")
    
    # Read and split the STL.
    mesh_obj = pv.read(input_stl)
    parts = mesh_obj.connectivity().split_bodies()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 parts in STL, found {len(parts)}")
    
    # Sort parts by x-centroid.
    x_centroids = [p.center[0] for p in parts]
    sorted_indices = np.argsort(x_centroids)
    labels = ["slat", "main", "flap"]
    
    for label, idx in zip(labels, sorted_indices):
        part = parts[idx].extract_surface()
        out_file = os.path.join(case_folder, f"{label}{case_number}.stl")
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(out_file)
        writer.SetInputData(part)
        writer.SetFileTypeToBinary()
        writer.Write()
        # print(f"[ok] wrote {out_file}")


# FFD Generation Functions
def _rotation_matrix(le, te):
    """2-D rotation that aligns the chord with +x."""
    dx, dy = te[0] - le[0], te[1] - le[1]
    theta  = np.arctan2(dy, dx)
    c, s   = np.cos(-theta), np.sin(-theta)
    return np.array([[c, -s], [s, c]])


def _rotate_xy(pts, R, origin):
    """Rotate xy of (N,3) pts about *origin* by 2×2 matrix R."""
    xy = pts[:, :2] - origin
    out = pts.copy()
    out[:, :2] = xy @ R.T + origin
    return out


def generate_global_ffd(baseline_stl="baseline30P30N.stl", ffd_filename="airfoilFFD.xyz", dims=None, margin=0.01):
    """
    Builds a simple 3-block FFD lattice around baseline_stl and writes a Tecplot-style .xyz file.
    
    dims : dict {"slat":(nx,ny,nz), "main":..., "flap":...}
           defaults to (6,3,2) for each.
    """
    if dims is None:
        dims = {name: (6, 3, 2) for name in ("slat", "main", "flap")}

    geom   = detect_geometry_info(baseline_stl)
    blocks = []
    dims_out = []

    for name in ("slat", "main", "flap"):
        pts  = geom[name]["mesh"].points
        le2d, te2d = geom[name]["LE_point"][:2], geom[name]["TE_point"][:2]

        R    = _rotation_matrix(le2d, te2d)
        ptsR = _rotate_xy(pts, R, le2d)

        nx, ny, nz = dims[name]
        x_min, x_max = ptsR[:, 0].min() - margin, ptsR[:, 0].max() + margin
        y_min, y_max = ptsR[:, 1].min() - margin, ptsR[:, 1].max() + margin
        z_min, z_max = ptsR[:, 2].min() - margin, ptsR[:, 2].max() + margin

        X = np.linspace(x_min, x_max, nx)
        Y = np.linspace(y_min, y_max, ny)
        Z = np.linspace(z_min, z_max, nz)
 
        grid = np.zeros((nx, ny, nz, 3))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, z in enumerate(Z):
                    grid[i, j, k] = (x, y, z)

        # rotate back to global coords
        flat = grid[..., :2].reshape(-1, 2)
        back = _rotate_xy(np.hstack((flat, np.zeros((flat.shape[0], 1)))), R.T, le2d)[:, :2]
        grid[..., :2] = back.reshape(nx, ny, nz, 2)

        blocks.append(grid)
        dims_out.append((nx, ny, nz))

    # write Tecplot-style .xyz
    with open(ffd_filename, "w") as f:
        f.write(f"{len(blocks)}\n")
        for nx, ny, nz in dims_out:
            f.write(f"{nx} {ny} {nz}\n")
        for block in blocks:
            for dim in range(3):  # x, y, z sweeps
                for k in range(block.shape[2]):
                    for j in range(block.shape[1]):
                        for i in range(block.shape[0]):
                            f.write(f"{block[i, j, k, dim]:.6f} ")
                        f.write("\n")

    print(f"FFD written at '{ffd_filename}'")
