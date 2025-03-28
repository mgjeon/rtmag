# RTMAG

This repository contains the source code for the paper "Real-time Extrapolation of Nonlinear Force-Free Fields from Photospheric Vector Magnetic Fields Using a Physics-Informed Neural Operator" (ApJS, accepted).

## Example

Caution: The reliability of AI-generated NLFFFs should be further investigated in future studies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgjeon/rtmag/blob/main/examples/example_colab.ipynb)

## Environment
- Python 3.11
- [PyTorch](https://pytorch.org/)
- [Neural Operator](https://github.com/NeuralOperator/neuraloperator)

```
conda env create -f environment.yml
conda activate rtmag
pip install "neuraloperator<1.0"
pip install -r https://raw.githubusercontent.com/NeuralOperator/neuraloperator/main/requirements.txt
pip install -e .
```

## Data

### [Low Lou (1990) NLFFF](https://ui.adsabs.harvard.edu/abs/1990ApJ...352..343L/abstract)

```
python scripts/dataset_lowlou.py --test_path <test-path> --train_path <train-path>
```

After running the above command, manually select and move 5000 files for training set and 1000 files for test set.

```
lowlou
├── test
│   └── case1.npz
├── train_5000
│   ├── b_0.150_0.014.npz
│   ├── b_0.150_0.041.npz
│   └── ...
└── val_1000
    ├── b_0.150_0.081.npz
    ├── b_0.150_0.343.npz
    └── ...
```


### [ISEE NLFFF database](https://hinode.isee.nagoya-u.ac.jp/nlfff_database/)

Download ISEE NLFFF and put the files in the following directory structure.

```
isee_data
├── 11078
│   └── 11078_20100609_000000.nc
├── 11089
│   └── 11089_20100725_000000.nc
│   ...
└── 12975
    ├── 12975NRT_20220327_120000.nc
    ├── 12975NRT_20220327_130000.nc
    └── ...
```

```
python scripts/dataset_isee.py --data_path <data-path> --dataset_path <dataset-path>
```

After running the above command, the files will be saved in the following directory structure.
```
isee_dataset
├── 11078
│   ├── input
│   │   └── input_11078_20100609_000000.npz
│   └── label
│       └── label_11078_20100609_000000.npz
├── 11089
│   ├── input
│   │   └── input_11089_20100725_000000.npz
│   └── label
│       └── label_11089_20100725_000000.npz
|   ...
└── 12975
    ├── input
    │   ├── input_12975NRT_20220327_120000.npz
    │   ├── input_12975NRT_20220327_130000.npz
    │   └── ...
    └── label
        ├── label_12975NRT_20220327_120000.npz
        ├── label_12975NRT_20220327_130000.npz
        └── ...
```

## Train

### Low Lou (1990) NLFFF
```
python train.py --config config/train_lowlou.json
```

### ISEE NLFFF
```
python train.py --config config/train.json
```

### Logs
```
tensorboard --logdir=<base_path>
```

## Acknowledgements
- I heavily referenced the [NF2 code](https://github.com/RobertJaro/NF2) for various tasks, such as the calculation of Low and Lou fields, metrics, and NLFFFs from PINNs. Thanks to the authors.
- I use data and code from the [ISEE NLFFF database](https://hinode.isee.nagoya-u.ac.jp/nlfff_database/) to create NLFFF datasets for real solar active regions. Thanks to the authors.
- I use [streamtracer](https://github.com/sunpy/streamtracer) and refer to the code from [flhtools](https://github.com/antyeates1983/flhtools) for field line tracing and plotting. Thanks to the authors.
- I use [pyvista](https://github.com/pyvista/pyvista) for 3D field line visualization in Colab notebook. Thanks to the authors.
- I appreciate the community efforts in creating other open-source packages used in this research, such as [SunPy](https://github.com/sunpy/sunpy) and [Astropy](https://github.com/astropy/astropy).
