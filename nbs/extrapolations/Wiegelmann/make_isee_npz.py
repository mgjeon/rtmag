import argparse
from rtmag.process.paper.utils import load_yaml_to_dict, get_input
from rtmag.process.wie.wiegelmann_utils import save_npz_from_bottom

def main(args):
    config_dict = load_yaml_to_dict(args.config)
    input_file = config_dict['input_file']
    bottom = get_input(input_file)

    # (Nx, Ny, 3)
    save_npz_from_bottom(bottom, 'bx_by_bz.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the simulation')
    args = parser.parse_args()
    main(args)

