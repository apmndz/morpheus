import os, time
import numpy as np
import matplotlib.pyplot as plt
from pygeo import DVGeometry
from pyspline import Curve
import trimesh
from geo_utils import *
from sampler import random_sample

def r_slat(v, g):
    """Apply slat rotation relative to horizontal if enabled"""
    g.rot_z["slat"].coef[:] = v[0] - g.init_slat_angle

def r_flap(v, g):
    """Apply flap rotation relative to horizontal if enabled"""
    g.rot_z["flap"].coef[:] = - v[0] - g.init_flap_angle

def tr_slat(v, g):
    """Apply slat translation if enabled. v contains [gap, overhang, vertical]"""
    C = g.extractCoef("slat")
    # Only use overhang for x movement, ignore gap parameter
    C[:,0] += v[0]  # v[1] is overhang
    C[:,1] += v[1]  # v[2] is vertical
    g.restoreCoef(C, "slat")

def tr_flap(v, g):
    """Apply flap translation if enabled. v contains [gap, overhang, vertical]"""
    C = g.extractCoef("flap")
    # Only use overhang for x movement, ignore gap parameter
    C[:,0] -= v[0]  # v[1] is overhang
    C[:,1] += v[1]  # v[2] is vertical
    g.restoreCoef(C, "flap")

def get_angle_wrt_x_axis(vec):
    return np.arctan2(vec[1], vec[0])*180/np.pi

def make_ffd_all(afoil_mesh, ffd_filename, margin = 0.005, shape=(6,3,2)):
    elts = decompose(afoil_mesh)
    with open(ffd_filename, "w") as f:
        f.write(f"{len(elts)}\n")
        nx, ny, nz = shape
        f.write(f"{nx} {ny} {nz}\n"*3)
        for elt in elts:
            grid = make_ffd_grid(elt, shape)
            for dim in range(3):           # x, then y, then z
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            f.write(f"{grid[i, j, k, dim]:.6f} ")
                        f.write("\n")
    print(f"FFD written at '{ffd_filename}'")

def make_ffd_grid(elt, shape, margin=0.005):
    le, te = get_le_te(elt.vertices)
    dx, dy = te[0] - le[0], te[1] - le[1]
    theta = np.arctan2(dy, dx)
    c, s = np.cos(-theta), np.sin(-theta)
    rot_mtx = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shifted_pts = elt.vertices - np.append(le, 0)
    rot_pts = shifted_pts @ rot_mtx.T
    xyz = []
    for i in range(3):
        coord_min, coord_max = rot_pts[:, i].min() - margin, rot_pts[:, i].max() + margin
        coords = np.linspace(coord_min, coord_max, shape[i])
        xyz.append(coords)
    grid = np.stack(np.meshgrid(*xyz, indexing='ij'), axis=-1).reshape(-1, 3)
    res = grid @ rot_mtx + np.append(le, 0)
    return res.reshape(*shape, 3)

def make_dvgeo(afoil_mesh, ffd_filename):
    dv = DVGeometry(ffd_filename)
    slat, _, flap = decompose(afoil_mesh)
    s_le, s_te = get_le_te(slat.vertices[slat.vertices[:, 2] == 0][:,:2])
    f_le, f_te = get_le_te(flap.vertices[flap.vertices[:, 2] == 0][:,:2])
    s_chrd_dir = s_te-s_le
    f_chrd_dir = f_te-f_le
    slat_angle = get_angle_wrt_x_axis(s_chrd_dir)
    flap_angle = get_angle_wrt_x_axis(f_chrd_dir)
    dv.addRefAxis("slat", Curve(x=[s_te[0]]*2, y=[s_te[1]]*2, z=[0, 0.1], k=2), axis="z", volumes=[0])
    dv.addRefAxis("flap", Curve(x=[f_le[0]]*2, y=[f_le[1]]*2, z=[0, 0.1], k=2), axis="z", volumes=[2])
    dv.init_slat_angle = slat_angle
    dv.init_flap_angle = flap_angle
    slat_trans = [0, 0]
    flap_trans = [0, 0]
    dv.addGlobalDV("tw_slat", [0], r_slat)
    dv.addGlobalDV("tw_flap", [0], r_flap)
    dv.addGlobalDV("tr_slat", slat_trans, tr_slat)
    dv.addGlobalDV("tr_flap", flap_trans, tr_flap)
    dv.addPointSet(afoil_mesh.vertices, "airfoil")
    dv.faces = afoil_mesh.faces
    return dv

def visualize(name, bounds, afoil):
    # bounds[0] (tuple): x bounds; bounds[1] (tuple): y bounds
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_title(name)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    loops = offset(afoil, 0)
    for loop in loops:
        ax.plot(loop[:, 0], loop[:, 1], 'b-', lw=1)
    return fig, ax

def apply_deformation(dvgeo, vals, min_gap=0.005, max_gap=None):
    """
    Generates an stl file of the input geometry in an output directory after applying 
    geometric deformations

    Args:
        dvgeo (DVGeometry): DVGeometry of the default multi-element airfoil configuration
        vals (dict): must have entries "tw_slat", "tw_flap", "tr_slat", "tr_flap"
            - tw_flap (float): flap rotation angle in degrees
            - tr_flap (np.array): dx, dy of the flap translation
            - tw_slat (float): slat rotation angle in degrees
            - tr_slat (np.array): dx, dy of the slat translation
        basename (str): default name of the multi-element airfoil. Includes NACA replacement digits if used
        out_path (str): path of desired output
        min_gap (float): minimum distance for main/slat & main/flap gaps
        vis (bool): True if images of the generated airfoil should be stored in "out_path/images".
                    Else, value of x,y bounds for the visualizations
    
    Returns:
        None | str: Returns None if properly generated, str with error message if invalid
    """
    f_r = vals["tw_flap"]
    f_t = vals["tr_flap"]
    s_r = vals["tw_slat"]
    s_t = vals["tr_slat"]
    dv = {"tw_slat": [s_r],
          "tw_flap": [f_r],
          "tr_slat": s_t,
          "tr_flap": f_t}
    dvgeo.setDesignVars(dv)
    afoil = dvgeo.update("airfoil")
    afoil_mesh = trimesh.Trimesh(afoil, dvgeo.faces)
    slat, main, flap = [Polygon(x) for x in offset(afoil_mesh, 0)][0:3]

    gap_slat = main.distance(slat)
    if gap_slat < min_gap:
        return f"slat gap below minimum (gap_slat: {gap_slat:.4f} < {min_gap})"
    elif max_gap is not None and gap_slat > max_gap:
        return f"slat gap above maximum: (gap_slat: {gap_slat:.4f} > {max_gap})"

    gap_flap = main.distance(flap)
    if gap_flap < min_gap:
        return f"flap gap below minimum (gap_flap: {gap_flap:.4f} < {min_gap})"
    elif max_gap is not None and gap_flap > max_gap:
        return f"flap gap above maximum (gap_flap: {gap_flap:.4f} > {max_gap})"
    return afoil_mesh

def generate_geometries(params, output="output"):
    """
    Randomly samples multiple geometric configurations of some default multi-element airfoil
    and stores them as stl files in an output directory.

    Args:
        def_stl_name (str): filename of the default stl
        bounds (dict): dictionary of the boundary values for geometric deformations.
                       Must have entries "tw_slat", "tw_flap", "of_slat", "of_flap" with
                       values as an np.array of 2 floats as lower and upper bounds
        NACA (bool | list[str]): if using NACA replacement, list of the 4/5 digits with last 2 digits combined
        output_dir (str): name of directory for stl file output
        vis (bool | str): if visualizing, name of output dir for photos

    Returns:
        None
    """
    start = time.time()
    def_stl_path = params["default_stl_path"]
    output_dir = output
    def_stl_obj = trimesh.load_mesh("input/"+def_stl_path)
    basename = def_stl_path.split("/")[-1][:-4]
    modified = replacement(
        def_stl_obj, 
        [params["slat"], params["main"], params["flap"]],
        [params["slat_chrd"], params["main_chrd"], params["flap_chrd"]])
    for name in ("slat", "main", "flap"):
        if params[name] is None:
            basename+="_def"
        elif isinstance(params[name], list):
            basename+="_"+"".join(map(str, params[name]))
        elif isinstance(params[name], str):
            basename+="_"+params[name].split("/")[-1][:-4]
        basename+="_"+str(params[name+"_chrd"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/STLs", exist_ok=True)
    os.makedirs(output_dir+"/images", exist_ok=True)
    ffd_file = output_dir+f"/ffd/{basename}FFD.xyz"
    if not os.path.exists(ffd_file):
        os.makedirs(output_dir+"/ffd", exist_ok=True)
        make_ffd_all(modified, ffd_file)
    else:
        print(f"FFD file '{ffd_file}' already exists. Using the existing grid")
    dv = make_dvgeo(modified, ffd_file)
    accepted = 0
    trials = 0
    continued_fail = 0
    n = params["n_samples"]
    while accepted < n:
        trial_time_start = time.time()
        s = random_sample(dv_ranges=params["bounds"])
        res = apply_deformation(dv, s, params["min_gap"], params["max_gap"])
        trials+=1
        if isinstance(res, str): # this means the gap failed
            print(res)
            continued_fail +=1
        else:
            name = f"{basename}_{s["tw_slat"]}_{s["tr_slat"][0]}_{s["tr_slat"][1]}_{s["tw_flap"]}_{s["tr_flap"][0]}_{s["tr_flap"][1]}"
            res.export(f"{output_dir}/STLs/{name}.stl")
            if accepted % params["vis"] == 0:
                fig, _ = visualize(name, [[-1, 2],[-0.5, 0.5]], res)
                fig.savefig(f"{output_dir}/images/{name}.png", dpi=300)
                plt.close(fig)
            print(f"Successful trial {trials} completed in {(time.time()-trial_time_start):.2f}s | {accepted}/{n} geometries")
            accepted+=1
            continued_fail = 0
        if continued_fail >= 20:
            print("Too many failures in a row. Try other bounds")
            return
    total_time = time.time()-start
    print(f"All operations completed successfully. Total time: {total_time:.2f}s")
    