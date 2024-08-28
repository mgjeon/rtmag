import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

#----------------------------------------------------------
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import numpy as np
from pathlib import Path
from rtmag.process.paper.utils import load_yaml_to_dict
from nf2.evaluation.unpack import load_cube

config_dict = load_yaml_to_dict(args.config)
filename = config_dict["filename"]

b = load_cube(filename, device=device, progress=True)

hmi_b_congrid = config_dict["input"]
bottom = np.load(hmi_b_congrid)
x = bottom['x']
y = bottom['y']

save_path = Path(config_dict["save_path"])
save_path.parent.mkdir(exist_ok=True, parents=True)
# folder_path = save_path / Path(hmi_b_congrid).parent.name
# folder_path.mkdir(exist_ok=True, parents=True)
# file = folder_path / "nf2.npz"

np.savez(save_path, b=b, x=x, y=y, z=y)
print(f"Saved to {save_path}")