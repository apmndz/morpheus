import os
import subprocess
import sys
import glob
import time
import argparse
from input.config import N_PROCESSORS


def run_command(cmd, cwd):
    """Run a shell command in the specified working directory with live output streaming."""
    print(f"Running: {cmd} in {cwd}")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=os.environ.copy()
    )

    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output:
            print(output.rstrip())
        if process.poll() is not None:
            break
    
    if process.returncode != 0:
        print(f"Error: Command '{cmd}' failed with return code {process.returncode}")
        sys.exit(process.returncode)
    return process

def mesh_case(cfd_dir):
    """Run the meshing commands one by one in the given CFD directory."""
    commands = [
        "blockMesh",
        "surfaceFeatureExtract",
        "decomposePar",
        f"mpirun -np {N_PROCESSORS} snappyHexMesh -parallel",
        "reconstructParMesh -constant -latestTime",
        "createPatch -overwrite"
    ]
    for cmd in commands:
        run_command(cmd, cwd=cfd_dir)
    # Convert the OpenFOAM mesh to VTK format.
    run_command("foamToVTK", cwd=cfd_dir)

def build_mesh_animation(output_dir):
    """Generate mesh visualizations and animations"""
    import pyvista as pv
    import imageio
    from geo_utils import detect_geometry_info
    from input.config import SAVE_EVERY
    
    pv.set_plot_theme('document')
    
    airfoil_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d))]
    
    for airfoil in airfoil_dirs:
        raw_basename = os.path.basename(airfoil)
        
        # Create visualization directory structure
        vis_dir = os.path.join(output_dir, raw_basename, "visualization")
        mesh_dir = os.path.join(vis_dir, "mesh")
        img_dir = os.path.join(mesh_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        frames_full = []
        frames_slat = []
        frames_flap = []
        
        cases_parent = os.path.join(airfoil, "cases")
        case_dirs = sorted([d for d in os.listdir(cases_parent)
                          if os.path.isdir(os.path.join(cases_parent, d))])
        
        for case in case_dirs:
            case_num = int(case.split('_')[1])
            
            # Skip if not a multiple of SAVE_EVERY
            if case_num % SAVE_EVERY != 0:
                continue
                
            cfd_dir = os.path.join(cases_parent, case, "CFD")
            vtk_file = os.path.join(cfd_dir, "VTK", "CFD_0", "boundary", "front.vtp")
            
            if not os.path.isfile(vtk_file):
                print(f"Warning: VTK file not found at {vtk_file}, skipping case {case}")
                continue
                
            print(f"Generating screenshots for mesh: {vtk_file}")
            
            try:
                mesh = pv.read(vtk_file)
                geom_info = detect_geometry_info(os.path.join(cfd_dir, "constant", "triSurface", "combined.stl"))
                
                # Image 1: Full view
                p1 = pv.Plotter(off_screen=True)
                p1.add_mesh(mesh, color="lightblue", opacity=1.0, show_edges=True)
                p1.view_vector((0, 0, 1))
                p1.reset_camera()
                focal_point = list(p1.camera.GetFocalPoint())
                focal_point[0] -= 1.4
                p1.camera.SetFocalPoint(focal_point)
                p1.camera.Zoom(22)
                p1.camera.Roll(90)
                p1.add_text(f"Case {case_num:03d}", position="upper_right", font_size=14)
                img_path = os.path.join(img_dir, f"mesh_{case_num:03d}_full.png")
                p1.show(screenshot=img_path)
                frames_full.append(imageio.imread(img_path))

                # Image 2: Zoom slat
                p2 = pv.Plotter(off_screen=True)
                p2.add_mesh(mesh, color="lightblue", opacity=1.0, show_edges=True)
                p2.view_vector((0, 0, 1))
                p2.reset_camera()
                le_point = geom_info["main"]["LE_point"]
                p2.camera.SetFocalPoint(le_point)
                p2.camera.Zoom(256)
                p2.camera.Roll(90)
                p2.add_text(f"Case {case_num:03d}", position="upper_right", font_size=14)
                img_path = os.path.join(img_dir, f"mesh_{case_num:03d}_slat.png")
                p2.show(screenshot=img_path)
                frames_slat.append(imageio.imread(img_path))

                # Image 3: Zoom flap
                p3 = pv.Plotter(off_screen=True)
                p3.add_mesh(mesh, color="lightblue", opacity=1.0, show_edges=True)
                p3.view_vector((0, 0, 1))
                p3.reset_camera()
                te_point = geom_info["main"]["TE_point"]
                p3.camera.SetFocalPoint(te_point)
                p3.camera.Zoom(256)
                p3.camera.Roll(90)
                p3.add_text(f"Case {case_num:03d}", position="upper_right", font_size=14)
                img_path = os.path.join(img_dir, f"mesh_{case_num:03d}_flap.png")
                p3.show(screenshot=img_path)
                frames_flap.append(imageio.imread(img_path))
                
            except Exception as e:
                print(f"Failed to process {vtk_file}: {e}")
                continue
        
        # Create animations for each view with loop=0 for infinite looping
        if frames_full:
            gif_path = os.path.join(mesh_dir, "meshes_full.gif")
            imageio.mimsave(gif_path, frames_full, fps=2, loop=0)
            
        if frames_slat:
            gif_path = os.path.join(mesh_dir, "meshes_slat.gif")
            imageio.mimsave(gif_path, frames_slat, fps=2, loop=0)
            
        if frames_flap:
            gif_path = os.path.join(mesh_dir, "meshes_flap.gif")
            imageio.mimsave(gif_path, frames_flap, fps=2, loop=0)
def main():
    parser = argparse.ArgumentParser(description="Mesh CFD cases or create animations for a specified airfoil")
    parser.add_argument("--airfoil", type=str, default="",
                        help="Name of the airfoil folder to process (inside 'output'). If not provided, all airfoil folders will be processed.")
    parser.add_argument("--only-animation", action="store_true",
                        help="Skip meshing and only build the mesh animation from existing VTK files.")
    args = parser.parse_args()

    output_dir = "output"
    if args.airfoil:
        airfoil_path = os.path.join(output_dir, args.airfoil)
        if not os.path.isdir(airfoil_path):
            print(f"Error: Airfoil folder '{args.airfoil}' not found in '{output_dir}'")
            sys.exit(1)
        airfoil_dirs = [airfoil_path]
    else:
        airfoil_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                        if os.path.isdir(os.path.join(output_dir, d))]
    
    if not args.only_animation:
        for airfoil in airfoil_dirs:
            cases_parent = os.path.join(airfoil, "cases")
            if not os.path.isdir(cases_parent):
                continue
            case_folders = sorted([d for d in os.listdir(cases_parent)
                                if os.path.isdir(os.path.join(cases_parent, d))])
            for case in case_folders:
                cfd_dir = os.path.join(cases_parent, case, "CFD")
                if os.path.isdir(cfd_dir):
                    print(f"\n--- Meshing case: {cfd_dir} ---")
                    mesh_case(cfd_dir)
                else:
                    print(f"Warning: CFD directory not found in {os.path.join(cases_parent, case)}")
        print("\nMeshing complete.")
    else:
        print("Skipping meshing. Using existing meshes.")

    # Build an animation composed of front view screenshots of each meshed case.
    build_mesh_animation(output_dir)

if __name__ == "__main__":
    main()

# command to run:
# python mesher.py --airfoil three_element_1 --only-animation