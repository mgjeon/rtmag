1. Install NF2

pip install git+https://github.com/RobertJaro/NF2.git

2. Run extrapolation (single)

python extrapolate.py --config config/nf2_11158.json

python extrapolate.py --config config/nf2_12673.json

3. Npz -> np.savez(file, b=b, x=x, y=y, z=y)

python unpack.py --config config/nf2_11158.yaml

python unpack.py --config config/nf2_12673.yaml