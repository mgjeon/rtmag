# rtmag

- Ubuntu 22.04
- Python 3.11
- CUDA 11.8
- [PyTorch](https://pytorch.org/)
- [Neural Operator](https://github.com/NeuralOperator/neuraloperator)

```
pip install -r requirements.txt
pip install -r https://raw.githubusercontent.com/NeuralOperator/neuraloperator/main/requirements.txt
pip install -e .
```

```
python train.py --config config/train.json
```

```
tensorboard --logdir=/mnt/d/models/uno
```

## nbs

```
pip install -r requirements-nbs.txt
pip install git+https://github.com/mgjeon/magnetic_field_line.git
```