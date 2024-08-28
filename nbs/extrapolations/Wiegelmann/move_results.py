import argparse
import shutil
from pathlib import Path
from rtmag.process.paper.utils import load_yaml_to_dict

def main(args):
    config_dict = load_yaml_to_dict(args.config)
    out_path = Path(config_dict['out_path'])
    out_path = out_path.parent

    out_path.mkdir(exist_ok=True, parents=True)
    shutil.move('allboundaries.dat', out_path/'allboundaries.dat')
    shutil.move('B0.bin', out_path/'B0.bin')
    shutil.move('B0.dat', out_path/'B0.dat')
    shutil.move('Bout.bin', out_path/'Bout.bin')
    shutil.move('Bout.dat', out_path/'Bout.dat')
    shutil.move('bx_by_bz.npz', out_path/'bx_by_bz.npz')
    shutil.move('grid.ini', out_path/'grid.ini')
    shutil.move('input.dat', out_path/'input.dat')
    shutil.move('Iteration_start.mark', out_path/'Iteration_start.mark')
    shutil.move('Iteration_stop.mark', out_path/'Iteration_stop.mark')
    shutil.move('prot.log', out_path/'prot.log')
    shutil.move('relax_20.log', out_path/'relax_20.log')
    shutil.move('relax_23.log', out_path/'relax_23.log')
    shutil.move('step.log', out_path/'step.log')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the simulation')
    args = parser.parse_args()
    main(args)





