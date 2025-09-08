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

def o_slat(v, g):
    """Apply slat translation if enabled. v contains [gap, overhang, vertical]"""
    C = g.extractCoef("slat")
    # Only use overhang for x movement, ignore gap parameter
    C[:,0] += v[0]  # v[1] is overhang
    C[:,1] += v[1]  # v[2] is vertical
    g.restoreCoef(C, "slat")

def o_flap(v, g):
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
    dv.addGlobalDV("of_slat", slat_trans, o_slat)
    dv.addGlobalDV("of_flap", flap_trans, o_flap)
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
    # ax.scatter(*afoil.vertices[:, :2].T, c="blue", s=1)
    loops = offset(afoil, 0)
    for loop in loops:
        ax.plot(loop[:, 0], loop[:, 1], 'b-', lw=1)
    return fig, ax

def apply_deformation(dvgeo, vals, basename, out_path, min_gap=0.005, vis=False):
    """
    Generates an stl file of the input geometry in an output directory after applying 
    geometric deformations

    Args:
        dvgeo (DVGeometry): DVGeometry of the default multi-element airfoil configuration
        vals (dict): must have entries "tw_slat", "tw_flap", "of_slat", "of_flap"
            - tw_flap (float): flap rotation angle in degrees
            - of_flap (np.array): dx, dy of the flap translation
            - tw_slat (float): slat rotation angle in degrees
            - of_slat (np.array): dx, dy of the slat translation
        basename (str): default name of the multi-element airfoil. Includes NACA replacement digits if used
        out_path (str): path of desired output
        min_gap (float): minimum distance for main/slat & main/flap gaps
        vis (bool): True if images of the generated airfoil should be stored in "out_path/images".
                    Else, value of x,y bounds for the visualizations
    
    Returns:
        None | str: Returns None if properly generated, str with error message if invalid
    """
    start = time.time()
    if vis:
        vis = [[-0.2, 1.4], [-0.4, 0.4]]
    if not os.path.exists(out_path+"/STLs"):
        os.makedirs(out_path+"/STLs")
    f_r = vals["tw_flap"]
    f_t = vals["of_flap"]
    s_r = vals["tw_slat"]
    s_t = vals["of_slat"]
    out_name = basename+"_"+"_".join([str(f_r), str(f_t[0]), str(f_t[1]), str(s_r), str(s_t[0]), str(s_t[1])])
    airfoil_output = os.path.join(out_path, "STLs", f"{out_name}.stl")
    dv = {"tw_slat": [s_r],
          "tw_flap": [f_r],
          "of_slat": s_t,
          "of_flap": f_t}
    dvgeo.setDesignVars(dv)
    afoil = dvgeo.update("airfoil")
    afoil_mesh = trimesh.Trimesh(afoil, dvgeo.faces)
    slat, main, flap = [Polygon(x) for x in offset(afoil_mesh, 0)]
    gap_slat = main.distance(slat)
    if gap_slat < min_gap:
        return f"Slat gap check failed (gap_slat: {gap_slat:.4f})."
    gap_flap = main.distance(flap)
    if gap_flap < min_gap:
        return f"Flap gap check failed (gap_flap: {gap_flap:.4f})."
    afoil_mesh.export(airfoil_output)
    if vis:
        vis_dir = out_path+"/images"
        fig, _ = visualize(out_name, vis, afoil_mesh)
        fig.savefig(vis_dir+f"/{out_name}.png", dpi=300)
        plt.close(fig)
    total_time = time.time()-start
    print(f"Valid geometry created. Time: {total_time:.2f}")
    
def mass_sample(def_stl_path, bounds, min_gap=0.005, n=300, main_replace=False, replace_chord_l=None, output_dir="output", vis=False):
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
    output_dir = "output/" + output_dir
    def_stl_obj = trimesh.load_mesh("input/"+def_stl_path)
    basename = def_stl_path.split("/")[-1][:-4]
    if main_replace:
        if isinstance(main_replace, list):
            # this means it's a naca airfoil
            mod_name = basename+"_"+"".join(map(str, main_replace))
            replacement = naca_generator(main_replace)
        elif isinstance(main_replace, str):
            # custom airfoil given by points
            mod_name = basename+"_"+main_replace.split("/")[-1][:-4]
            replacement = read_dat("input/"+main_replace)
        else:
            raise ValueError(f"main_replace should only be of type (False, list, or str). It's currently of type {type(main_replace)}")
        modified = main_replacement(def_stl_obj, replacement, replace_chord_l)
    else:
        modified = def_stl_obj
        mod_name = basename+"_main"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/STLs", exist_ok=True)
    os.makedirs(output_dir+"/images", exist_ok=True)
    ffd_file = output_dir+f"/ffd/{mod_name}FFD.xyz"
    if not os.path.exists(ffd_file):
        os.makedirs(output_dir+"/ffd", exist_ok=True)
        make_ffd_all(modified, ffd_file)
    else:
        print(f"FFD file '{ffd_file}' already exists. Usingthe existing grid")
    dv = make_dvgeo(modified, ffd_file)
    accepted = 0
    trials = 0
    continued_fail = 0
    while accepted < n:
        print(f"Initiating trial {trials}.")
        sample = random_sample(dv_ranges=bounds)
        res = apply_deformation(dv, sample, mod_name, output_dir, min_gap, vis)
        trials+=1
        if res is None:
            accepted+=1
            print(f"Trial {trials} completed. {accepted}/{n} geometries generated")
            continued_fail = 0
        else:
            print(str(res))
            continued_fail +=1
        if continued_fail >= 10:
            print("Too many failures in a row. Try other bounds")
            return
    total_time = time.time()-start
    print(f"All operations completed successfully. Total time: {total_time:.2f}s")

bounds_1 = {"tw_slat": [0, 60], "tw_flap": [-90, 90], "of_slat": [[-0.03, -0.02], [0.02, 0.02]], "of_flap": [[-0.03, -0.02], [0.07, 0.01]]}

# for filename in os.listdir("coord_seligFmt"):
#     file_path = os.path.join("coord_seligFmt", filename)
#     mass_sample("input/30P30N.stl", def_bounds, n=1, output_dir="UIUC3", main_replace=file_path, vis=True)

# main element replace with dat file defining points
mass_sample("30P30N.stl", bounds_1, n=100, output_dir="output7", main_replace='coord_seligFmt/ag16.dat', replace_chord_l=1, vis=True)

# main element replace with a naca airfoil
mass_sample("30P30N.stl", bounds_1, n=100, output_dir="output8", main_replace=[0, 0, 12], replace_chord_l=1, vis=True)