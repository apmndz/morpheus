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


def detect_geometry_info(airfoil_stl):
    """Return dict[name] → {LE_point, TE_point, mesh, chord_vector}.
    
    Args:
        airfoil_stl (numpy-stl.mesh.Mesh): stl object of the multi-element airfoil

    """
    parts = numpy_stl_to_pyvista(airfoil_stl).connectivity().split_bodies()
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
            "chord_vector": (te - le) # / np.linalg.norm(te - le),
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

def scale_to_unit(airfoil_stl, dst):
    g  = detect_geometry_info(airfoil_stl); 
    le, te = g["main"]["LE_point"], g["main"]["TE_point"]
    s = 1.0/np.linalg.norm(te - le)
    m = airfoil_stl
    m.vectors[:] = ((m.vectors.reshape(-1,3)-le)*s).reshape(-1,3,3)
    os.makedirs(os.path.dirname(dst), exist_ok=True); m.save(dst)
    print(f"scaled chord to 1 m, scale {s:.4f})")

def stretch_transform(airfoil_stl, output_stl, new_thickness=0.5):
    # Load the STL mesh
    m = airfoil_stl
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


def generate_global_ffd(baseline_stl=mesh.Mesh.from_file("input/baseline30P30N.stl"), ffd_filename="airfoilFFD.xyz", dims=None, margin=0.001):
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

# NACA Generation Functions

def thickness_dist(t, x, CTE = True):
    if CTE:
        a = -0.1036
    else:
        a = -0.1015
    return 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 + a*x**4)

def camber_line(params, x):
    assert np.all(np.logical_and(x >= 0, x <= 1)), 'Found x > 1 or x < 0'
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)

    if len(params) == 2:
        m = params[0]/100
        p = params[1]/10

        if p == 0:
            dy_c = -2*m*x
            return y_c, dy_c
        elif p == 1:
            dy_c = 2*m*(1 - x)
            return y_c, dy_c

        mask1 = np.logical_and(x >= 0, x < p)
        mask2 = np.logical_and(x >= p, x <= 1)
        y_c[mask1] = (m/p**2)*(2*p*x[mask1] - x[mask1]**2)
        dy_c[mask1] = (2*m/p**2)*(p - x[mask1])
        y_c[mask2] = (m/(1 - p)**2)*((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)
        dy_c[mask2] = (2*m/(1 - p)**2)*(p - x[mask2])

    elif len(params) == 3:
        l, p, q = params
        c_l, x_f = 3/20*l, p/20

        f = lambda x: x*(1 - np.sqrt(x/3)) - x_f
        df = lambda x: 1 - 3*np.sqrt(x/3)/2
        old_m = 0.5
        cond = True
        while cond:
            new_m = np.max([old_m - f(old_m)/df(old_m), 0])
            cond = (np.abs(old_m - new_m) > 1e-15)
            old_m = new_m        
        m = old_m
        r = (3*m - 7*m**2 + 8*m**3 - 4*m**4)/np.sqrt(m*(1 - m)) - 3/2*(1 - 2*m)*(np.pi/2 - np.arcsin(1 - 2*m))
        k_1 = c_l/r

        mask1 = np.logical_and(x >= 0, x <= m)
        mask2 = np.logical_and(x > m, x <= 1)
        if q == 0:            
            y_c[mask1] = k_1*((x[mask1]**3 - 3*m*x[mask1]**2 + m**2*(3 - m)*x[mask1]))
            dy_c[mask1] = k_1*(3*x[mask1]**2 - 6*m*x[mask1] + m**2*(3 - m))
            y_c[mask2] = k_1*m**3*(1 - x[mask2])
            dy_c[mask2] = -k_1*m**3*np.ones_like(dy_c[mask2])

        elif q == 1:
            k = (3*(m - x_f)**2 - m**3)/(1 - m)**3
            y_c[mask1] = k_1*((x[mask1] - m)**3 - k*(1 - m)**3*x[mask1] - m**3*x[mask1] + m**3)
            dy_c[mask1] = k_1*(3*(x[mask1] - m)**2 - k*(1 - m)**3 - m**3)
            y_c[mask2] = k_1*(k*(x[mask2] - m)**3 - k*(1 - m)**3*x[mask2] - m**3*x[mask2] + m**3)
            dy_c[mask2] = k_1*(3*k*(x[mask2] - m)**2 - k*(1 - m)**3 - m**3)

        else:
            raise ValueError('Q must be 0 for normal camber or 1 for reflex camber.')

    else:
        raise ValueError('The first input must be a tuple of the 2 or 3 digits that represent the camber line.')   

    return y_c, dy_c

def naca_generator(params, nb_samples = 400, scale = 1, origin = (0, 0), cosine_spacing = True, verbose = True, CTE = True):
    if len(params) == 3:
        params_c = params[:2]
        t = params[2]/100
        if verbose:
            print(f'Generating naca M = {params_c[0]}, P = {params_c[1]}, XX = {t*100}')
    elif len(params) == 4:
        params_c = params[:3]
        t = params[3]/100
        if verbose:
            print(f'Generating naca L = {params_c[0]}, P = {params_c[1]}, Q = {params_c[2]}, XX = {t*100}')
    else:
        raise ValueError('The first argument must be a tuple of the 4 or 5 digits of the airfoil.')    

    if cosine_spacing:
        beta = np.pi*np.linspace(1, 0, nb_samples + 1, endpoint = True)
        x = (1 - np.cos(beta))/2
    else:
        x = np.linspace(1, 0, nb_samples + 1, endpoint = True)

    y_c, dy_c = camber_line(params_c, x)
    y_t = thickness_dist(t, x, CTE)
    theta = np.arctan(dy_c)
    x_u = x - y_t*np.sin(theta)
    x_l = x + y_t*np.sin(theta)
    y_u = y_c + y_t*np.cos(theta)
    y_l = y_c - y_t*np.cos(theta)
    x = np.concatenate([x_u, x_l[:-1][::-1]], axis=0)
    y = np.concatenate([y_u, y_l[:-1][::-1]], axis=0)
    pos = np.stack([
            x*scale + origin[0],
            y*scale + origin[1]
        ], axis=-1
    )
    pos[0], pos[-1] = np.array([1, 0]), np.array([1, 0])
    return pos

# NACA Replacement Functions

def make_naca_stl(naca_digits, le, te):
    """Creates an stl object of a NACA airfoil

    Args:
        naca_digits (list): NACA 4 or 5 digits, combine last 2 digits (length = 3 or 4)
        le (np.array): x,y coordinates of the leading edge
        te (np.array): x,y coordinates of the trailing edge

    Returns:
        mesh.Mesh: a numpy-stl object of the NACA airfoil
    """
    naca_geo = naca_generator(naca_digits)
    chord_l = np.linalg.norm(le-te)
    scaled = chord_l*naca_geo
    theta = np.arcsin((te[1]-le[1]) / chord_l)
    rot_mtx = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = scaled @ rot_mtx.T
    translated = rotated+np.array(le)
    N = translated.shape[0]
    duplicated = np.vstack([translated, translated])  # Now (2N, 2)
    z_values = np.hstack([
        np.zeros(N),      # First N points at z = 0
        0.1 * np.ones(N)  # Next N points at z = 0.1
    ])
    # Stack x, y, z together
    coords_3d = np.hstack([duplicated, z_values[:, np.newaxis]])
    naca_mesh = make_mesh(coords_3d)
    Xmod_reshaped = naca_mesh.reshape(-1, 3, 3)
    stl_obj = mesh.Mesh(np.zeros(Xmod_reshaped.shape[0], dtype=mesh.Mesh.dtype))
    stl_obj.vectors[:] = Xmod_reshaped
    return stl_obj

def remove_main(afoil_geo):
    """ Removes the main body of a given multi-element-airfoil. Converts to pyVista to find
    desired component to remove and then converts back to stl.

    Args:
        afoil_geo (mesh.Mesh): a numpy-stl object of the multi-element-airfoil

    Returns:
        mesh.Mesh: a numpy-stl object of the multi-element-airfoil without the main element
    """
    # og_pv_mesh = numpy_stl_to_pyvista(afoil_geo)
    # parts = og_pv_mesh.connectivity().split_bodies()
    # if len(parts) != 3:
    #     raise ValueError(f"Expected 3 parts in STL, found {len(parts)}")
    # x_centroids = [p.center[0] for p in parts]
    # sorted_indices = np.argsort(x_centroids)
    # parts.pop(sorted_indices[1])
    elts = detect_geometry_info(afoil_geo)
    modified_pv_mesh = pv.PolyData()
    for name, elt in elts.items():
        if name != "main":
            modified_pv_mesh = modified_pv_mesh.merge(elt["mesh"])
    return pyvista_to_numpy_stl(modified_pv_mesh)

def make_mesh(geo):
    N = geo.shape[0]/2
    lower = np.arange(0, N - 1)
    upper = lower + N
    tri1 = np.stack([lower, lower + 1, upper], axis=1)
    tri2 = np.stack([lower + 1, upper + 1, upper], axis=1)
    faces = np.vstack([tri1, tri2]).astype(int)
    triangular_faces = []
    # Loop over each face and get the actual coordinates from coords_3d
    for face in faces:
        points = geo[face]  # Get the actual points (coordinates) for each face
        triangular_faces.append(points)
    # Now triangular_faces contains the actual 3D points for each triangular face
    triangular_faces = np.array(triangular_faces)
    reshaped = triangular_faces.reshape(-1,3)
    return reshaped

def numpy_stl_to_pyvista(stl_mesh_obj):
    """Converts an stl.mesh.Mesh object to a pyvista.PolyData object."""
    # Extract vertices and faces
    # stl.mesh.Mesh.vectors gives (N, 3, 3) array: N triangles, each with 3 (x,y,z) vertices
    # pyvista.PolyData expects (M, 3) for vertices and (F, 3) for faces (indices into vertices)

    # Flatten all vertices and find unique ones
    all_verts = stl_mesh_obj.vectors.reshape(-1, 3)
    unique_verts, inverse_indices = np.unique(all_verts, axis=0, return_inverse=True)

    # Reconstruct faces using indices into unique_verts
    faces = inverse_indices.reshape(-1, 3) # Now faces refer to indices in unique_verts

    # PyVista requires faces to be flattened with a leading '3' for each triangle
    # e.g., [3, v0_idx, v1_idx, v2_idx, 3, v3_idx, v4_idx, v5_idx, ...]
    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()

    return pv.PolyData(unique_verts, pv_faces)

def pyvista_to_numpy_stl(pv_mesh_obj):
    """Converts a pyvista.PolyData object back to an stl.mesh.Mesh object."""
    if isinstance(pv_mesh_obj, pv.UnstructuredGrid):
        pv_mesh_obj = pv_mesh_obj.extract_surface()

    vertices = pv_mesh_obj.points
    # PyVista faces format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
    # We need to reshape to (N, 3) for numpy-stl
    faces = pv_mesh_obj.faces.reshape(-1, 4)[:, 1:]

    stl_data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face_indices in enumerate(faces):
        stl_data['vectors'][i] = vertices[face_indices]

    return mesh.Mesh(stl_data, remove_duplicate_polygons=True, calculate_normals=True)

def make_naca_replacement(default_stl, naca_digits, chord_l=None):
    elts = detect_geometry_info(default_stl)
    le, te = elts["main"]["LE_point"][:2], elts["main"]["TE_point"][:2]
    og_te_x = te[0]
    if chord_l is not None:
        te[0] = le[0]+(chord_l**2-(te[1]-le[1])**2)**(1/2)  # change TE x value to make chord length
    naca = make_naca_stl(naca_digits, le, te)
    flap_slat = remove_main(default_stl)
    modified_pv_mesh = pv.PolyData()
    for name, elt in elts.items():
        if name == "flap":
            modified_pv_mesh = modified_pv_mesh.merge(elt["mesh"].translate((te[0]-og_te_x, 0, 0)))
        elif name != "main":
            modified_pv_mesh = modified_pv_mesh.merge(elt["mesh"])
    flap_slat = pyvista_to_numpy_stl(modified_pv_mesh)
    combined_data = np.concatenate([flap_slat.data, naca.data])
    return mesh.Mesh(combined_data)