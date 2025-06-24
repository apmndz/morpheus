# Input/Output settings
RAW_STL = "input/STLs/baseline30P30N.stl"
OUT_DIR = "output"
N_SAMPLES = 50

# Geometry constraints 
MIN_GAP = 0.0065
MAX_GAP = 0.03

# Visualization settings
DPI = 240
DOT_SIZE = 0.000001  # scatter plot marker size
SAVE_IMAGES = True    # turn previews on/off
SAVE_EVERY = 1       # save PNG every Nth accepted sample
SCATTER_STEP = 2     # plot every k-th vertex
ANIM_FPS = 10        # animation frames per second

# Design variable ranges
TWIST_SLAT = (-90, 90)
TWIST_FLAP = (-90, 90)
GAP_SLAT = (MIN_GAP, MAX_GAP)
GAP_FLAP = (MIN_GAP, MAX_GAP)
OVR_SLAT = (-0.03, 0.02) # overhang
# OVR_SLAT = (0,0)

OVR_FLAP = (-0.08, 0.07)
# OVR_FLAP = (2, 2)
# OVR_FLAP = (0,0)


VERT_SLAT = (-0.02, 0.02)  # vertical movement range for slat
# VERT_SLAT = (0, 0)  # vertical movement range for slat

VERT_FLAP = (-0.02, 0.01)  # vertical movement range for flap
# VERT_FLAP = (0.0, 0.0)  # vertical movement range for flap


# Design Features Toggle
ENABLE_BUMPS = False  # Toggle random surface bumps on/off
ENABLE_SLAT_ROTATION = False # Toggle slat rotation
ENABLE_FLAP_ROTATION = True   # Toggle flap rotation
ENABLE_SLAT_TRANSLATION = True # Toggle slat translation
ENABLE_FLAP_TRANSLATION = False # Toggle flap translation

# Bump parameters
BUMP_ITER_RANGE = (1, 5)      # number of iterations
BUMP_RANGE = (-5e-3, 5e-3)    # bump magnitude range
BUMP_SIZE_RANGE = (2, 6)      # number of points to bump

# Compile design variable ranges
DV_RANGES = {
    "tw_slat": TWIST_SLAT,
    "tw_flap": TWIST_FLAP,
    "of_slat": [GAP_SLAT, OVR_SLAT, VERT_SLAT],  # Keep GAP_SLAT for constraints
    "of_flap": [GAP_FLAP, OVR_FLAP, VERT_FLAP]   # Keep GAP_FLAP for constraints
}

# OpenFOAM settings
N_PROCESSORS = 10  # Number of processors for parallel operations

# Sampling methods
MAIN_DV_SAMPLING = "random"      # Method for sampling slat/flap transformations: "random", "lhs", "sobol", etc.
LOCAL_BUMP_SAMPLING = "random" # Method for sampling local bumps: "random", "lhs", "sobol", etc.
 