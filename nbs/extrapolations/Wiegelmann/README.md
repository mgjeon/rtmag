# Structure

z=0 -> `allboundaries.dat` -> `B0.bin` (Initial Field, usually potential field) -> `Bout.bin` (Force-free field)


# Commands

## 0. Compile

bash clean.sh

bash compile.sh

bash compile_openmp.sh

## 1. ISEE z=0 => `bx_by_bz.npz`

python make_isee_npz.py --config config/11158.yaml

python make_isee_npz.py --config config/12673.yaml

## 2. `bx_by_bz.npz` => `input.dat` & `grid.ini`

python create_input.py

## 3. Preprocessing `input.dat` => `allboundaries.dat`

bash preprocessing.sh

## 4. Calculate potential field only (B0.bin)

bash relax.sh 23 10000                           

or

bash relax_openmp.sh 23 10000

## 5. Calculate nonlinear force-free field (Bout.bin)

bash relax.sh 20 10000

## 6. Move results

python move_results.py --config config/11158.yaml

python move_results.py --config config/12673.yaml

## 7. unpack

python unpack_bin.py --config config/11158.yaml

python unpack_bin.py --config config/12673.yaml