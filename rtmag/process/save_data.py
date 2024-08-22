import os 
from tqdm import tqdm
from rtmag.process.load_isee import load_fields
import numpy as np
from joblib import Parallel, delayed

def do_job(file, input_dir, label_dir):
    filename = os.path.basename(file).split('.')[0]
    input_path = os.path.join(input_dir, f'input_{filename}.npz')
    label_path = os.path.join(label_dir, f'label_{filename}.npz')
    if (not os.path.exists(input_path)) and (not os.path.exists(label_path)):
        x, y, z, b, bp = load_fields(file)
        dx, dy, dz = np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z))
        
        input = b[:, :, :, 0][:, :, :, None]
        np.savez(input_path, input=input, dx=dx, dy=dy, dz=dz)

        # label = b[:, :, :, :50]
        # pot = bp[:, :, :, :50]
        label = b 
        pot = bp
        np.savez(label_path, label=label, pot=pot, x=x, y=y, z=z)

def save_input_label(file_list, data_dir):
    input_dir = os.path.join(data_dir, 'input')
    label_dir = os.path.join(data_dir, 'label')

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    Parallel(n_jobs=4)(delayed(do_job)(file, input_dir, label_dir) for file in tqdm(file_list))