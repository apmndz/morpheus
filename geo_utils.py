import numpy as np
from sklearn.decomposition import PCA
from shapely.geometry import Polygon
import trimesh
from trimesh.transformations import transform_points

def get_le_te(afoil):
    """
    Returns the x,y coordinates of the leading edge and trailing edge

    Args:
        afoil (np.array): set of points that define the outline of the airfoil. Can be either 3D or 2D points
    
    Returns:
        np.array (N,2): leading edge of the airfoil
        np.array (N,2): trailing edge of the airfoil
        np.array (N,2): chord direction from leading edge to trailing edge
    """
    if afoil.shape[1] == 3:
        afoil = afoil[np.isclose(afoil[:, 2], 0.0)][:, :2]
    elif afoil.shape[1] != 2:
        raise ValueError(f"The airfoil should be of dimension (N,2). This airfoil is of dimesion {afoil.shape}")
    center = np.mean(afoil, axis=0)
    centered_points = afoil - center
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    chord_dir = pca.components_[0]
    projections = centered_points @ chord_dir
    i_le = np.argmin(projections)
    i_te = np.argmax(projections)
    leading_edge = afoil[i_le]
    trailing_edge = afoil[i_te]
    return leading_edge, trailing_edge

def extrude(afoil2D, l=0.1):
    """
    Generates an stl object of the extruded point set

    Args:
        afoil2D (np.array(N,2)): list of points that define the 2D airfoil
        l (float): extrusion distance

    Returns:
        Trimesh: a Trimesh object of the resultant extruded airfoil
    """
    if not np.allclose(afoil2D[0], afoil2D[-1]):
        afoil2D = np.vstack([afoil2D, afoil2D[0]])
    poly = Polygon(afoil2D).buffer(0)
    extruded = trimesh.creation.extrude_polygon(poly, height=l)
    return extruded

def read_dat(path):
    replacement = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                prts = line.split(" ")
                x_str, y_str = prts[0], prts[-1]
                x, y = float(x_str), float(y_str)
                replacement.append(np.array([x, y]))
            except ValueError:
                continue
    return np.array(replacement)

def offset(afoil_mesh, margin=0.005):
    """
    Offsets the boundary of a all given airfoil components in a mesh 
    (assuming it is an extruded airfoil).

    Args:
        afoil_mesh (Trimesh): mesh of the multi-element airfoil
        margin (float): distance of the offset

    Returns:
        np.array[np.array]: array of offset(s) in order based on path traversal
    """
    plane_origin = [0, 0, 0]
    plane_normal = [0, 0, 1]
    section = afoil_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        raise ValueError("The slicing plane did not intersect the mesh.")
    # Convert to 2D path
    path_2d, T = section.to_2D()
    paths = path_2d.discrete
    paths.sort(key=lambda tup: tup[:, 0].min())
    offset_paths = []
    for segment in paths:
        poly = Polygon(np.array(segment))
        offsetted = np.array(poly.buffer(margin).exterior.coords)
        offset_3d = np.column_stack([offsetted, np.zeros(len(offsetted))])
        original_3d = transform_points(offset_3d, T)
        offset_paths.append(original_3d[:, :2])
    return offset_paths

def bounding_box(mesh, margin=0.005):
    """
    Compute an axis-aligned bounding box (AABB) with a margin for a trimesh mesh.

    Parameters:
        mesh (Trimesh): input mesh
        margin (float): amount to expand the bounding box on each side

    Returns:
        bounds (np.ndarray): 2x3 array of min/max bounds
    """
    bounds = mesh.bounds  # shape: (2, 3) → [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_corner = bounds[0] - margin
    max_corner = bounds[1] + margin
    return np.vstack([min_corner, max_corner])

def decompose(afoil_mesh):
    """
    Removes the main element of a multi-element airfoil configuration

    Args:
        afoil2D (Trimesh): mesh of the multi-element airfoil

    Returns:
        (Trimesh, Trimesh): (main element, flap and slat elements)
    """
    components = afoil_mesh.split(only_watertight=True)
    components.sort(key=lambda c: c.centroid[0])
    # if len(components) != 3:
    #     raise ValueError(f"Expected 3 parts. Found {len(components)} parts.")
    return components

def main_replacement(default_mesh, replace_afoil_2D, chord_l=None):
    """
    Replaces the main element of a multi-element airfoil configuration with
    a given 2D airfoil and chord length. If no chord length is given, the length
    in the x direction of the main element is preserved. The replacement airfoil will have
    an angle of attack of 0 degrees.

    Args:
        default_mesh (Trimesh): mesh of the default multi-element airfoil
        replace_afoil_2D (np.array): set of points that outline the 2D airfoil
        chord_l (float): chord length of the replacement airfoil
    
    Returns:
        Trimesh: mesh of the replaced multi-element airfoil
    """
    slats, main, flaps = decompose(default_mesh)
    le, te = get_le_te(main.vertices)
    deltaH = [0, le[1]-te[1], 0]
    og_xchrd = te[0]-le[0]
    if not chord_l:
        chord_l = og_xchrd
    else:
        deltaH[0] += chord_l-og_xchrd
    flaps.apply_translation(deltaH)
    scaled_afoil_2D = chord_l*replace_afoil_2D
    replace_afoil_mesh = extrude(scaled_afoil_2D)
    replace_afoil_mesh.apply_translation([*le, 0])
    return trimesh.util.concatenate([slats, replace_afoil_mesh, flaps])

# NACA Generation (AirFRANS)

def thickness_dist(t, x, CTE=True):
    if CTE:
        a = -0.1036
    else:
        a = -0.1015
    return (
        5
        * t
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a * x**4)
    )

def camber_line(params, x):
    assert np.all(np.logical_and(x >= 0, x <= 1)), "Found x > 1 or x < 0"
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)

    if len(params) == 2:
        m = params[0] / 100
        p = params[1] / 10

        if p == 0:
            dy_c = -2 * m * x
            return y_c, dy_c
        elif p == 1:
            dy_c = 2 * m * (1 - x)
            return y_c, dy_c

        mask1 = np.logical_and(x >= 0, x < p)
        mask2 = np.logical_and(x >= p, x <= 1)
        y_c[mask1] = (m / p**2) * (2 * p * x[mask1] - x[mask1] ** 2)
        dy_c[mask1] = (2 * m / p**2) * (p - x[mask1])
        y_c[mask2] = (m / (1 - p) ** 2) * (
            (1 - 2 * p) + 2 * p * x[mask2] - x[mask2] ** 2
        )
        dy_c[mask2] = (2 * m / (1 - p) ** 2) * (p - x[mask2])

    elif len(params) == 3:
        l, p, q = params
        c_l, x_f = 3 / 20 * l, p / 20

        f = lambda x: x * (1 - np.sqrt(x / 3)) - x_f
        df = lambda x: 1 - 3 * np.sqrt(x / 3) / 2
        old_m = 0.5
        cond = True
        while cond:
            new_m = np.max([old_m - f(old_m) / df(old_m), 0])
            cond = np.abs(old_m - new_m) > 1e-15
            old_m = new_m
        m = old_m
        r = (3 * m - 7 * m**2 + 8 * m**3 - 4 * m**4) / np.sqrt(
            m * (1 - m)
        ) - 3 / 2 * (1 - 2 * m) * (np.pi / 2 - np.arcsin(1 - 2 * m))
        k_1 = c_l / r

        mask1 = np.logical_and(x >= 0, x <= m)
        mask2 = np.logical_and(x > m, x <= 1)
        if q == 0:
            y_c[mask1] = k_1 * (
                (x[mask1] ** 3 - 3 * m * x[mask1] ** 2 + m**2 * (3 - m) * x[mask1])
            )
            dy_c[mask1] = k_1 * (3 * x[mask1] ** 2 - 6 * m * x[mask1] + m**2 * (3 - m))
            y_c[mask2] = k_1 * m**3 * (1 - x[mask2])
            dy_c[mask2] = -k_1 * m**3 * np.ones_like(dy_c[mask2])

        elif q == 1:
            k = (3 * (m - x_f) ** 2 - m**3) / (1 - m) ** 3
            y_c[mask1] = k_1 * (
                (x[mask1] - m) ** 3
                - k * (1 - m) ** 3 * x[mask1]
                - m**3 * x[mask1]
                + m**3
            )
            dy_c[mask1] = k_1 * (3 * (x[mask1] - m) ** 2 - k * (1 - m) ** 3 - m**3)
            y_c[mask2] = k_1 * (
                k * (x[mask2] - m) ** 3
                - k * (1 - m) ** 3 * x[mask2]
                - m**3 * x[mask2]
                + m**3
            )
            dy_c[mask2] = k_1 * (3 * k * (x[mask2] - m) ** 2 - k * (1 - m) ** 3 - m**3)

        else:
            raise ValueError("Q must be 0 for normal camber or 1 for reflex camber.")

    else:
        raise ValueError(
            "The first input must be a tuple of the 2 or 3 digits that represent the camber line."
        )

    return y_c, dy_c

def naca_generator(
    params,
    nb_samples=400,
    scale=1,
    origin=(0, 0),
    cosine_spacing=True,
    verbose=True,
    CTE=True,
):
    if len(params) == 3:
        params_c = params[:2]
        t = params[2] / 100
        if verbose:
            print(f"Generating naca M = {params_c[0]}, P = {params_c[1]}, XX = {t*100}")
    elif len(params) == 4:
        params_c = params[:3]
        t = params[3] / 100
        if verbose:
            print(
                f"Generating naca L = {params_c[0]}, P = {params_c[1]}, Q = {params_c[2]}, XX = {t*100}"
            )
    else:
        raise ValueError(
            "The first argument must be a tuple of the 4 or 5 digits of the airfoil."
        )

    if cosine_spacing:
        beta = np.pi * np.linspace(1, 0, nb_samples + 1, endpoint=True)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(1, 0, nb_samples + 1, endpoint=True)

    y_c, dy_c = camber_line(params_c, x)
    y_t = thickness_dist(t, x, CTE)
    theta = np.arctan(dy_c)
    x_u = x - y_t * np.sin(theta)
    x_l = x + y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    y_l = y_c - y_t * np.cos(theta)
    x = np.concatenate([x_u, x_l[:-1][::-1]], axis=0)
    y = np.concatenate([y_u, y_l[:-1][::-1]], axis=0)
    pos = np.stack([x * scale + origin[0], y * scale + origin[1]], axis=-1)
    pos[0], pos[-1] = np.array([1, 0]), np.array([1, 0])
    return pos
