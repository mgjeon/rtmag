from pathlib import Path
from rtmag.process.load_isee import nc_list
from rtmag.process.save_data import save_input_label
import time

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    dataset_path = Path(args.dataset_path)

    entries = data_path.iterdir()
    directories = [entry for entry in entries if entry.is_dir()]
    print(len(directories))

    start = time.time()

    dataset_path.mkdir(exist_ok=True)

    for directory in directories:
        file_list = nc_list(directory)
        res_path = dataset_path / directory.name
        print(f'Number : {len(file_list)}')
        print(f'Save   : {res_path}')
        save_input_label(file_list, res_path)
        break

    runtime = time.time() - start

    print(f'Runtime: {runtime:.2f} sec')