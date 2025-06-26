# Input/Output settings
GEO_TYPE = "stl"    # "stl" or "naca"
RAW_STL = "input/STLs/three_element_1.stl"
NACA = "0012"   # if GEO_TYPE == "naca", use RAW_STL flaps & slats. REQUIRES a RAW_STL
OUT_DIR = "output"
N_SAMPLES = 250

# Geometry constraints
MIN_GAP = 0.02
MAX_GAP = 0.03

# Visualization settings
DPI = 240
DOT_SIZE = 0.000001  # scatter plot marker size
SAVE_IMAGES = True    # turn previews on/off
SAVE_EVERY = 5       # save PNG every Nth accepted sample
SCATTER_STEP = 2     # plot every k-th vertex
ANIM_FPS = 10        # animation frames per second

# Design variable ranges
TWIST_SLAT = (-10, 10)
TWIST_FLAP = (-10, 10)
GAP_SLAT = (MIN_GAP, MAX_GAP)
GAP_FLAP = (MIN_GAP, MAX_GAP)
OVR_SLAT = (-0.02, 0.05) # overhang
OVR_FLAP = (-0.02, 0.52)

# Bump parameters
BUMP_ITER_RANGE = (1, 5)      # number of iterations
BUMP_RANGE = (-5e-2, 5e-2)    # bump magnitude range
BUMP_SIZE_RANGE = (2, 6)      # number of points to bump

# Compile design variable ranges
DV_RANGES = {
    "tw_slat": TWIST_SLAT,
    "tw_flap": TWIST_FLAP,
    "of_slat": [GAP_SLAT, OVR_SLAT],
    "of_flap": [GAP_FLAP, OVR_FLAP]
}

# OpenFOAM settings
N_PROCESSORS = 10  # Number of processors for parallel operations

# Sampling methods
MAIN_DV_SAMPLING = "random"      # Method for sampling slat/flap transformations: "random", "lhs", "sobol", etc.
LOCAL_BUMP_SAMPLING = "random" # Method for sampling local bumps: "random", "lhs", "sobol", etc.
