"""
make_animation.py – stitch the images in ./output into a GIF or MP4.

Examples
--------
# quickest way (GIF at 10 fps):
python make_animation.py

# custom settings (MP4, 30 fps, JPEGs):
python make_animation.py --outfile animation.mp4 --fps 30 --pattern '*.jpg'
"""
import argparse
import glob
import os
import imageio.v2 as imageio

def build_animation(folder: str, pattern: str, fps: int, outfile: str) -> None:
    """Collect images -> save animation."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f'No files matched {os.path.join(folder, pattern)}')

    frames = [imageio.imread(p) for p in paths]

    if outfile.lower().endswith('.gif'):
        imageio.mimsave(outfile, frames, fps=fps, loop=0)  # loop=0 means infinite loop    
    else: 
        imageio.mimsave(
            outfile,
            frames,
            fps=fps,
            codec='libx264',            
            quality=8,                  # 0‑10 (higher → better/larger)
            macro_block_size=None       # stops imageio auto‑padding
        )
    print(f'wrote {outfile} with {len(frames)} frames at {fps} fps.')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder',  default='output',   help="where your images live")
    ap.add_argument('--pattern', default='*.png',    help="glob pattern to match (quotes!)")
    ap.add_argument('--fps',     default=10, type=int,
                    help="frames per second in the output video")
    ap.add_argument('--outfile', default='animation.gif',
                    help="name of the resulting .gif or .mp4")
    build_animation(**vars(ap.parse_args()))
