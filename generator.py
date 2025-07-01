import os, time, random, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stl import mesh
from scipy.spatial import KDTree
from shapely.geometry import Polygon
import glob
import fileinput
import argparse
from geo_utils import *
from pygeo           import DVGeometry
from pyspline        import Curve
from sampler import get_samples, random_sample
from animation import build_animation
from mesher import mesh_case, build_mesh_animation

# Import all settings from config
from config import *

def sample_bumps(n, bump_range, method=LOCAL_BUMP_SAMPLING):
    if method == "random":
        return [random.uniform(bump_range[0], bump_range[1]) for _ in range(n)]
    else:
        from sampler import get_samples
        bump_dv_range = {"bump": bump_range}
        samples = get_samples(method, n, dv_ranges=bump_dv_range)
        return [s["bump"] for s in samples]

def apply_local_bumps(dv, idx_f, idx_b, 
                      n_iterations_range=BUMP_ITER_RANGE, 
                      bump_range=BUMP_RANGE, 
                      bump_size_range=BUMP_SIZE_RANGE,
                      local_bump_method="random"):
    n_iter = random.randint(*n_iterations_range)
    for _ in range(n_iter):
        num_sel = random.randint(*bump_size_range)
        sel = np.random.choice(idx_f, num_sel, replace=False)
        bumps = np.array(sample_bumps(num_sel, bump_range, method=local_bump_method))
        dv["mainX"][sel] += bumps
        dv["mainX"][idx_b[np.isin(idx_f, sel)]] += bumps
        dv["mainY"][sel] += bumps
        dv["mainY"][idx_b[np.isin(idx_f, sel)]] += bumps

def mk_curve(p, z=[0,.1]): 
    return Curve(x=[p[0]]*2, y=[p[1]]*2, z=z, k=2)

def get_initial_angle(chord_vector):
    """Calculate angle between chord vector and horizontal (x-axis)"""
    return np.arctan2(chord_vector[1], chord_vector[0]) * 180/np.pi

def r_slat(v, g):
    """Apply slat rotation relative to horizontal if enabled"""
    g.rot_z["slat"].coef[:] = - v[0] - g.initial_slat_angle

def r_flap(v, g):
    """Apply flap rotation relative to horizontal if enabled"""
    g.rot_z["flap"].coef[:] = - v[0] - g.initial_flap_angle

def o_slat(v, g):
    """Apply slat translation if enabled. v contains [gap, overhang, vertical]"""
    if ENABLE_SLAT_TRANSLATION:
        C = g.extractCoef("slat")
        # Only use overhang for x movement, ignore gap parameter
        C[:,0] += v[1]  # v[1] is overhang
        C[:,1] += v[2]  # v[2] is vertical
        g.restoreCoef(C, "slat")

def o_flap(v, g):
    """Apply flap translation if enabled. v contains [gap, overhang, vertical]"""
    if ENABLE_FLAP_TRANSLATION:
        C = g.extractCoef("flap")
        # Only use overhang for x movement, ignore gap parameter
        C[:,0] -= v[1]  # v[1] is overhang
        C[:,1] += v[2]  # v[2] is vertical
        g.restoreCoef(C, "flap")

def make_dvgeo(pts, geom, ffd_filename):
    """Initialize DVGeometry with pre-calculated initial angles"""
    dv = DVGeometry(ffd_filename)
    
    # Calculate initial angles from geometry info we already have
    dv.initial_slat_angle = -90 + get_initial_angle(geom["slat"]["chord_vector"])
    dv.initial_flap_angle = get_initial_angle(geom["flap"]["chord_vector"])
    
    # Set up reference axes and design variables
    dv.addRefAxis("slat", mk_curve(geom["slat"]["TE_point"]), axis="z", volumes=[0])
    dv.addRefAxis("flap", mk_curve(geom["flap"]["LE_point"]), axis="z", volumes=[2])
    dv.addLocalDV("mainX", -0.8, .8, axis="x")
    dv.addLocalDV("mainY", -0.8, .8, axis="y")
    
    # Initialize angles based on enabled/disabled state
    # slat_init = [0.0] if ENABLE_SLAT_ROTATION else [dv.initial_slat_angle]
    # flap_init = [0.0] if ENABLE_FLAP_ROTATION else [dv.initial_flap_angle]
    
    # Initialize translations based on enabled/disabled state
    slat_trans = [0.0, 0.0, 0.0] if ENABLE_SLAT_TRANSLATION else None  # Added third component
    flap_trans = [0.0, 0.0, 0.0] if ENABLE_FLAP_TRANSLATION else None  # Added third component
    
    dv.addGlobalDV("tw_slat", [90.0], r_slat)
    dv.addGlobalDV("tw_flap", [0.0], r_flap)
    if slat_trans is not None:
        dv.addGlobalDV("of_slat", slat_trans, o_slat)
    if flap_trans is not None:
        dv.addGlobalDV("of_flap", flap_trans, o_flap)
    dv.addPointSet(pts, "airfoil")
    idx = dv.getLocalIndex(1,1)
    return dv, idx[:,:,0].ravel(), idx[:,:,1].ravel()

def update_decompose_par_dict(filepath, n_proc):
    """Update the numberOfSubdomains in decomposeParDict"""
    for line in fileinput.input(filepath, inplace=True):
        if line.strip().startswith('numberOfSubdomains'):
            print(f'numberOfSubdomains {n_proc};')
        else:
            print(line, end='')

def visualize(name, *pointSets):
    fig, ax = plt.subplots(figsize=(6, 3))
    for points in pointSets:
        xpad = (points[:,0].max()-points[:,0].min())*0.075
        ypad = (points[:,1].max()-points[:,1].min())*.7
        ax.scatter(points[:, 0], points[:, 1], s=1)
    ax.set_xlim(pointSets[0][:,0].min()-xpad, pointSets[0][:,0].max()+xpad)
    ax.set_ylim(pointSets[0][:,1].min()-ypad, pointSets[0][:,1].max()+ypad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(name)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y")
    return fig, ax

def generate_from_dvgeo(dvgeo, vals, basename, out_path, NACA=False, vis=False):
    """
    Generates an stl file of the input geometry in an output directory after applying 
    geometric deformations

    Args:
        dvgeo (pygeo.DVGeometry): DVGeometry of the default multi-element airfoil configuration
        vals (dict): must have entries "tw_slat", "tw_flap", "of_slat", "of_flap"
            - tw_flap (float): flap rotation angle in degrees
            - of_flap (np.array): dx,dy of the flap translation
            - tw_slat (float): slat rotation angle in degrees
            - of_slat (np.array): dx,dy of the slat translation
        basename (str): default name of the multi-element airfoil. Includes NACA replacement digits if used
        NACA (bool | list[str]): if using NACA replacement, list of the 4/5 digits with last 2 digits combined
        vis (bool): True if images of the generated airfoil should be stored in "out_path/images"
    
    Returns:
        None | str: Returns none if properly generated, str with error message if invalid
    """
    
    start = time.time()
    f_r = vals["tw_flap"]
    f_t = vals["of_flap"]
    s_r = vals["tw_slat"]
    s_t = vals["of_slat"]
    out_name = f"{basename}_{f_r}_({f_t[0]},{f_t[1]})_{s_r}_({s_t[0]},{s_t[1]})"
    airfoil_output = os.path.join(out_path, "STLs", f"{out_name}.stl")

    dv = dict()
    dv["tw_slat"] = [s_r]
    dv["tw_flap"] = [f_r]
    dv["of_slat"] = s_t
    dv["of_flap"] = f_t

    dvgeo.setDesignVars(dv)
    afoil = dvgeo.update("airfoil")
    afoil_stl = afoil.reshape(-1,3,3)
    stl_obj = mesh.Mesh(np.zeros(afoil_stl.shape[0], dtype=mesh.Mesh.dtype))
    stl_obj.vectors[:] = afoil_stl

    try:
        geom = detect_geometry_info(stl_obj)

        loop_main = boundary_loop_2d(geom["main"]["mesh"])
        poly_main = Polygon(loop_main)

        loop_slat = boundary_loop_2d(geom["slat"]["mesh"])
        poly_slat = Polygon(loop_slat)

        loop_flap = boundary_loop_2d(geom["flap"]["mesh"])
        poly_flap = Polygon(loop_flap)
        
        # buf_main = poly_main.buffer(MIN_GAP)

        gap_slat = poly_main.distance(poly_slat)
        gap_flap = poly_main.distance(poly_flap)
    except Exception as e:
        return e

    if gap_slat < MIN_GAP:
        return f"Slat gap check failed (gap_slat: {gap_slat:.4f})."
    elif gap_flap < MIN_GAP:
        return f"Flap gap check failed (gap_flap: {gap_flap:.4f})."
    
    stl_obj.save(airfoil_output)

    if vis:
        vis_dir = os.path.join(out_path, "images")
        fig, ax = visualize(out_name, afoil, dvgeo.FFD.coef)
        # slat_le = geom["slat"]["LE_point"]
        # ax.scatter(slat_le[0], slat_le[1], s=1, color="red")
        fig.savefig(os.path.join(vis_dir, f"{out_name}.png"), dpi=300)
        plt.close(fig)
    
    total_time = time.time() - start
    if NACA:
        print(f"Valid geometry created, NACA={"".join(map(str, NACA))} & [{f_r}, {f_t[:2]}, {s_r}, {s_t[:2]}]. Time: {total_time:2f}s")
    else:
        print(f"Valid geometry created, [{f_r}, {f_t}, {s_r}, {s_t}]. Time: {total_time:2f}s")

def sample_generate_geometries(def_stl_name, bounds, n=300, NACA = False, NACA_chord_l = None, output_dir="output", vis=False):
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
    # if not all(["tw_slat", "tw_flap", "of_slat", "of_flap"] in bounds):
    #     raise ValueError('"tw_slat", "tw_flap", "of_slat", "of_flap" keys not in "bounds"')
    start = time.time()
    print(start)
    stl_input = "input/"+def_stl_name
    def_stl_obj = mesh.Mesh.from_file(stl_input)
    out_dir = os.path.join(os.getcwd(), output_dir)
    basename = def_stl_name[:-4]
    if NACA:
        basename += "".join(map(str, NACA))
        def_stl_obj = make_naca_replacement(def_stl_obj, NACA, NACA_chord_l)
    os.makedirs(out_dir, exist_ok=True)
    ffd_file = os.path.join(out_dir, "ffd", f"{basename}FFD.xyz")
    os.makedirs(out_dir+"/STLs", exist_ok=True)
    if not os.path.exists(ffd_file):
        os.makedirs(out_dir+"/ffd", exist_ok=True)
        generate_global_ffd(def_stl_obj, ffd_filename=ffd_file)
    else:
        print(f"FFD file '{ffd_file}' already exists. Using the existing grid.")
    if vis:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    geom0 = detect_geometry_info(def_stl_obj)
    pts0  = def_stl_obj.vectors.reshape(-1, 3)
    dvgeo, idx_f, idx_b = make_dvgeo(pts0, geom0, ffd_file)
    accepted = 0
    trials = 0
    continued_fail = 0
    while accepted < n:
        print(f"Initiating trial {trials}.")
        sample = random_sample(dv_ranges=bounds)
        res = generate_from_dvgeo(dvgeo, sample, basename, out_dir, NACA, vis=vis)
        trials+=1
        if res is None:
            accepted+=1
            print(f"Trial {trials} completed. {accepted}/{n} geometries generated")
            continued_fail = 0
        else:
            print(str(res))
            continued_fail +=1
        if continued_fail >= 5:
            print("Too many failures in a row. Try other bounds")
            return
    total_time = time.time()-start
    print(f"All operations completed successfully. Total time: {total_time:.2f}s")

def generate(geo_only=False, mesh_only=False, mesh_vis_only=False):
    t_start = time.time()
    
    # Create a descriptive suffix based on all enabled features
    config_suffix = []
    if ENABLE_SLAT_ROTATION:
        config_suffix.append("slatRot")
    if ENABLE_FLAP_ROTATION:
        config_suffix.append("flapRot")
    if ENABLE_SLAT_TRANSLATION:
        config_suffix.append("slatTrans")
    if ENABLE_FLAP_TRANSLATION:
        config_suffix.append("flapTrans")
    if ENABLE_BUMPS:
        config_suffix.append("bumps")
    
    # Combine all suffixes
    config_str = "_".join(config_suffix) if config_suffix else "baseline"
    sampling_suffix = f"_{MAIN_DV_SAMPLING}_{LOCAL_BUMP_SAMPLING}"
    
    raw_basename = os.path.splitext(os.path.basename(RAW_STL))[0]
    airfoil_output = os.path.join(OUT_DIR, f"{raw_basename}_{config_str}{sampling_suffix}")
    
    # Geometry generation
    if not mesh_only and not mesh_vis_only:
        # Geometry generation code
        os.makedirs(airfoil_output, exist_ok=True)
        
        # Create geometry visualization folders
        vis_dir = os.path.join(airfoil_output, "visualization")
        geom_dir = os.path.join(vis_dir, "geometry")
        images_dir = os.path.join(geom_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        ffd_file = os.path.join(airfoil_output, f"{raw_basename}FFD.xyz")
        
        SCALED = os.path.join(airfoil_output, "scaled.stl")
        scale_to_unit(RAW_STL, SCALED)
        
        if not os.path.exists(ffd_file):
            generate_global_ffd(baseline_stl=SCALED, ffd_filename=ffd_file)
        else:
            print(f"FFD file '{ffd_file}' already exists. Using the existing grid.")
        
        geom0 = detect_geometry_info(SCALED)
        pts0  = mesh.Mesh.from_file(SCALED).vectors.reshape(-1, 3)
        dvgeo, idx_f, idx_b = make_dvgeo(pts0, geom0, ffd_file)
        base = {k: v.copy() for k, v in dvgeo.getValues().items()}
        
        if SAVE_IMAGES:
            xpad = (pts0[:,0].max()-pts0[:,0].min())*0.075
            ypad = (pts0[:,1].max()-pts0[:,1].min())*.7
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_aspect("equal")
            ax.set_xlim(pts0[:,0].min()-xpad, pts0[:,0].max()+xpad)
            ax.set_ylim(pts0[:,1].min()-ypad, pts0[:,1].max()+ypad)
            cloud  = ax.scatter([], [], s=DOT_SIZE)
            l_main, = ax.plot([], [], '-',  lw=.6, color='gray')
            l_slat, = ax.plot([], [], '-',  lw=.6, color='steelblue')
            l_flap, = ax.plot([], [], '-',  lw=.6, color='seagreen')
            l_buf,  = ax.plot([], [], '--', lw=.6, color='orange')
            title = ax.set_title("")
        
        acc_rows, rej_rows = [], []
        accepted = 0
        trials = 0
        
        # Continue iterating until the number of accepted samples reaches N_SAMPLES.
        while accepted < N_SAMPLES:
            iter_start = time.time()
            trials += 1
            
            # Generate a single DV sample using configured method
            sample = get_samples(method=MAIN_DV_SAMPLING, n=1, dv_ranges=DV_RANGES)[0]
            dv = {k: v.copy() for k, v in base.items()}
            dv["tw_slat"][0] = sample["tw_slat"]
            dv["tw_flap"][0] = sample["tw_flap"]
            if ENABLE_SLAT_TRANSLATION and "of_slat" in dv:
                dv["of_slat"][:] = sample["of_slat"]
            if ENABLE_FLAP_TRANSLATION and "of_flap" in dv:
                dv["of_flap"][:] = sample["of_flap"]
    
            # Only apply bumps if enabled in config
            if ENABLE_BUMPS:
                apply_local_bumps(dv, idx_f, idx_b)

            dvgeo.setDesignVars(dv)
            pts_mod = dvgeo.update("airfoil")
        
            tmp = os.path.join(airfoil_output, "tmp.stl")
            m = mesh.Mesh(np.zeros(len(pts_mod) // 3, mesh.Mesh.dtype))
            m.vectors[:] = pts_mod.reshape(-1, 3, 3)
            m.save(tmp)
        
            try:
                geom = detect_geometry_info(tmp)
                loop_main = boundary_loop_2d(geom["main"]["mesh"])
                poly_main = Polygon(loop_main)
        
                loop_slat = boundary_loop_2d(geom["slat"]["mesh"])
                poly_slat = Polygon(loop_slat)
        
                loop_flap = boundary_loop_2d(geom["flap"]["mesh"])
                poly_flap = Polygon(loop_flap)
                
                buf_main = poly_main.buffer(MIN_GAP)
        
                gap_slat = poly_main.distance(poly_slat)
                gap_flap = poly_main.distance(poly_flap)
            except Exception as e:
                error_msg = f"Iteration {trials} failed: {str(e)}"
                print(error_msg)
                rej_rows.append({"trial": trials, "reason": str(e)})
                continue
        
            if gap_slat < MIN_GAP or gap_flap < MIN_GAP:
                error_msg = (f"Iteration {trials} failed: Gap check failed "
                             f"(gap_slat: {gap_slat:.4f}, gap_flap: {gap_flap:.4f}).")
                print(error_msg)
                rej_rows.append({"trial": trials, "reason": "Gap check failed"})
                continue
        
            if SAVE_IMAGES and accepted % SAVE_EVERY == 0:
                cloud.set_offsets(pts_mod[::SCATTER_STEP, :2])
                l_main.set_data(loop_main.T)
                l_slat.set_data(loop_slat.T)
                l_flap.set_data(loop_flap.T)
                l_buf.set_data(*buf_main.exterior.xy)
                title.set_text(f"Sample {accepted:03d}")
                img_path = os.path.join(images_dir, f"geo_{accepted:03d}.png")
                fig.savefig(img_path, dpi=DPI)
        
            # Process the valid sample.
            combined_stl = os.path.join(airfoil_output, "combined.stl")
            os.rename(tmp, combined_stl)
        
            stretched_stl = os.path.join(airfoil_output, "stretched_combined.stl")
            stretch_transform(combined_stl, stretched_stl, new_thickness=0.5)
            os.remove(combined_stl)
        
            # Create a top-level "cases" folder within the airfoil output directory.
            cases_parent = os.path.join(airfoil_output, "cases")
            os.makedirs(cases_parent, exist_ok=True)
        
            # Now, for the accepted sample, create its subfolder inside the "cases" folder:
            case_folder = os.path.join(cases_parent, f"case_{accepted:03d}")
            os.makedirs(case_folder, exist_ok=True)
            
            # Copy CFD basecase folder into this case subfolder using rsync or shutil.copytree.
            source_cfd = os.path.join("input", "CFD_basecase")
            dest_cfd = os.path.join(case_folder, "CFD")
            if os.path.exists(dest_cfd):
                shutil.rmtree(dest_cfd)
            shutil.copytree(source_cfd, dest_cfd)
            
            # Update the decomposeParDict with the number of processors from config
            decompose_par_dict = os.path.join(dest_cfd, "system", "decomposeParDict")
            update_decompose_par_dict(decompose_par_dict, N_PROCESSORS)
        
            # Define the destination folder for the STL parts.
            # Our STL files will go into: case_folder/CFD/constant/triSurface
            triSurface_folder = os.path.join(dest_cfd, "constant", "triSurface")
            if not os.path.exists(triSurface_folder):
                os.makedirs(triSurface_folder, exist_ok=True)
        
            # Call the splitting function to output the parts into triSurface_folder.
            split_stl_case(stretched_stl, triSurface_folder, accepted)
            os.remove(stretched_stl)
        
            # Remove any appended numbers in the file names.
            import glob
            def rename_stl_files(folder):
                mappings = {
                    "flap": "flap.stl",
                    "slat": "slat.stl",
                    "main": "main.stl",
                    "combined": "combined.stl"
                }
                for prefix, new_name in mappings.items():
                    files = glob.glob(os.path.join(folder, f"{prefix}*.stl"))
                    for f in files:
                        target = os.path.join(folder, new_name)
                        os.rename(f, target)
            rename_stl_files(triSurface_folder)
        
            # Modify the logging section to handle disabled translations
            acc_rows.append({
                "idx": accepted,
                "cmd_gap_slat": dv["of_slat"][0] if "of_slat" in dv else None,
                "cmd_gap_flap": dv["of_flap"][0] if "of_flap" in dv else None,
                "gap_slat": gap_slat,
                "gap_flap": gap_flap
            })
            iteration_time = time.time() - iter_start
            print(f"Accepted sample {accepted:03d}  (iteration time: {iteration_time:.3f}s)")
            accepted += 1
        
        # Write CSV logs in the airfoil output folder.
        pd.DataFrame(acc_rows).to_csv(os.path.join(airfoil_output, "design_variables.csv"), index=False)
        if rej_rows:
            pd.DataFrame(rej_rows).to_csv(os.path.join(airfoil_output, "rejected_samples.csv"), index=False)
        
        print("done —", accepted, "accepted |", len(rej_rows), "rejected")
        print("Total elapsed time: {:.3f}s".format(time.time() - t_start))
        
        # After processing all cases…
        # Create geometry animation
        gif_file = os.path.join(geom_dir, "geometries.gif")
        build_animation(folder=images_dir, pattern='geo_*.png', fps=ANIM_FPS, outfile=gif_file)
        
        # Discard the scaled.stl file once the script is finished.
        if os.path.exists(SCALED):
            os.remove(SCALED)
    
    # Meshing
    if not geo_only and not mesh_vis_only:  # Changed this condition
        if mesh_only:
            print("\nStarting mesh generation for all cases...")
            t_mesh_start = time.time()
        
        # Process all cases in the airfoil's cases directory
        cases_parent = os.path.join(airfoil_output, "cases")
        case_folders = sorted([d for d in os.listdir(cases_parent)
                             if os.path.isdir(os.path.join(cases_parent, d))])
        
        for case in case_folders:
            cfd_dir = os.path.join(cases_parent, case, "CFD")
            if os.path.isdir(cfd_dir):
                print(f"\n--- Meshing case: {cfd_dir} ---")
                mesh_case(cfd_dir)
            else:
                print(f"Warning: CFD directory not found in {os.path.join(cases_parent, case)}")
        
        if mesh_only:
            mesh_time = time.time() - t_mesh_start
            print(f"\nMeshing complete. Time elapsed: {mesh_time:.2f}s")
            return
    
    # Mesh visualization
    if not geo_only and (mesh_vis_only or not mesh_only):
        print("Building mesh animations...")
        build_mesh_animation(OUT_DIR)
    
    total_time = time.time() - t_start
    print(f"\nAll operations completed successfully. Total time: {total_time:.2f}s")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Airfoil geometry generation and meshing tool')
    # with open('bounds.yaml', 'r') as f: # hyperparameters of the model
    #     params = yaml.safe_load(f)
    
    # parser.add_argument('--geo-only', action='store_true',
    #                     help='Only generate geometry and geometry visualizations')
    # parser.add_argument('--mesh-only', action='store_true',
    #                     help='Only perform mesh generation')
    # parser.add_argument('--mesh-vis-only', action='store_true',
    #                     help='Only generate mesh visualizations')

    # args = parser.parse_args()
    
    # # Ensure only one flag is set at a time
    # if sum([args.geo_only, args.mesh_only, args.mesh_vis_only]) > 1:
    #     parser.error("Please specify only one operation mode")
    
    # generate(
    #     geo_only=args.geo_only,
    #     mesh_only=args.mesh_only,
    #     mesh_vis_only=args.mesh_vis_only
    # )

    bounds = {"tw_slat": [0,60], 
              "tw_flap": [-90, 90],
              "of_slat": [[-0.03, -0.02, 0], [0.02, 0.02, 0]], 
              "of_flap": [[-0.08, -0.02, 0], [0.07, 0.01, 0]]}
    
    sample_generate_geometries("baseline30P30N.stl", bounds, 10, output_dir="output2", vis=False)
    sample_generate_geometries("baseline30P30N.stl", bounds, 5, [2,4,12], output_dir="output", vis=False)
    sample_generate_geometries("baseline30P30N.stl", bounds, 5, [6,4,12], 1, output_dir="output2", vis=False)
    # python generator.py
    # python generator.py --geo-only
    # python generator.py --mesh-only
    # python generator.py --mesh-vis-only