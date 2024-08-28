import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

#----------------------------------------------------------
import numpy as np
from pathlib import Path
from rtmag.process.paper.utils import load_yaml_to_dict, load_bin

config_dict = load_yaml_to_dict(args.config)
nx = config_dict["nx"]
ny = config_dict["ny"]
nz = config_dict["nz"]

out_path = Path(config_dict["out_path"])
save_out_path = Path(config_dict["save_out_path"])

out = load_bin(out_path, nx, ny, nz).astype(np.float32)

hmi_b_congrid = config_dict["hmi"]
bottom = np.load(hmi_b_congrid)
x = bottom['x']
y = bottom['y']

save_out_path.parent.mkdir(exist_ok=True, parents=True)
np.savez(save_out_path, b=out, x=x, y=y, z=y)
print(f"Saved to {save_out_path}")

