import yaml
import os
from generator import generate_geometries

with open('config.yaml', 'r') as file:
    params = yaml.safe_load(file)

# generate_geometries(params, "output")

# generating with many main element replacements

# for filename in os.listdir("input/coord_seligFmt"):
#     file_path = os.path.join("coord_seligFmt", filename)
#     params["main_replace"] = file_path
#     generate_geometries(params, "output1")