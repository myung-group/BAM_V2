# BAM-NequIP

A JAX/Flax implementation of NequIP (Neural Equivariant Interatomic Potentials) using the Flax NNX API and e3nn-jax for equivariant operations.

## Overview

A modified version of NequIP is an E(3)-equivariant graph neural network architecture for learning interatomic potentials. This implementation provides:

- **Flax NNX API**: Modern, Pythonic neural network implementation
- **e3nn-jax**: Efficient equivariant operations using spherical harmonics
- **JIT compilation**: Full JAX JIT support for fast training and inference
- **Scan-based training**: Optional `jax.lax.scan` for reduced Python overhead
- **EMA weights**: Exponential moving average for stable training
- **separated layer normalization**: to stabilize training

## Features

- E(3)-equivariant message passing with spherical harmonics
- Bessel radial basis functions with polynomial cutoff
- Per-species embeddings and skip connections
- Separated Layer normalization for equivariant features
- Huber loss for robust training
- Learning rate scheduling with reduce-on-plateau
- Checkpoint saving/loading

## Installation

```bash
# Clone the repository
git clone https://github.com/myung-group/BAM-nequip.git
cd BAM-nequip

# Install dependencies
pip install -e .
```

### Requirements

- Python >= 3.12
- JAX >= 0.8.0
- Flax >= 0.12.0
- e3nn-jax >= 0.20.8
- jraph >= 0.0.6
- optax >= 0.2.5
- ASE >= 3.27.0
- matscipy >= 1.2.0

## Usage

### Training

1. Prepare your dataset in extended XYZ format with energy and forces.
2. Create a configuration file (see `input_nequip.json` for an example):

```json
{
    "cutoff": 5.0,
    "train_path": "data/train.xyz",
    "atomic_numbers": [1, 6, 7, 8],
    "atom_energies": [-13.587, -1029.489, -1484.981, -2041.982],
    "hidden_size": 128,
    "lmax": 3,
    "n_layers": 5,
    "batch_size": 5,
    "n_epochs": 1000,
    "learning_rate": 0.01
}
```

1. Run training:

```bash
python -m bam_3bpa.train_3bpa
```

### Model Architecture

The model consists of:

1. **Species embedding**: Linear projection of one-hot encoded atomic species
2. **Interaction layers**: Equivariant convolutions with:
  - Separated Linear transformation of node features
  - Tensor product with spherical harmonics of edge vectors
  - Radial MLP for distance-dependent weights
  - Gated nonlinearity
  - Per-species skip connections
3. **Readout**: Linear projections to scalar per-atom energies

### Configuration Options


| Parameter           | Description                      | Default |
| ------------------- | -------------------------------- | ------- |
| `cutoff`            | Radial cutoff distance (Å)       | 5.0     |
| `hidden_size`       | Hidden feature dimension         | 128     |
| `lmax`              | Maximum angular momentum         | 3       |
| `n_layers`          | Number of interaction layers     | 5       |
| `radial_basis_size` | Number of Bessel basis functions | 8       |
| `radial_mlp_size`   | Hidden size of radial MLP        | 64      |
| `radial_mlp_layers` | Number of radial MLP layers      | 3       |
| `learning_rate`     | Initial learning rate            | 0.01    |
| `energy_weight`     | Loss weight for energy           | 1.0     |
| `force_weight`      | Loss weight for forces           | 1.0     |
| `use_scan`          | Use jax.lax.scan for training    | true    |
| `grad_clip`         | Gradient clipping norm           | 1.0     |
| `ema_decay`         | EMA decay rate                   | 0.99    |


### Comparison with other codes

In order to verify correctness of the implementation, we compare performance on
the 3BPA dataset to two different PyTorch NequIP implementations: 

1. The results from Musaelian et al. in ["Learning local equivariant

representations for large-scale atomistic
dynamics"]([https://www.nature.com/articles/s41467-023-36329-y](https://www.nature.com/articles/s41467-023-36329-y)), using a version
of their `[nequip](https://github.com/mir-group/nequip)` repository.  
2. The
results from Batatia et al. in ["MACE: Higher Order Equivariant Message
Passing Neural Networks for Fast and Accurate Force
Fields"](https://arxiv.org/abs/2206.07697), using a version of their `[mace](https://github.com/ACEsuit/mace)` repository.

We use the same hyperparameters as [1], which can be viewed in `[input/input_3bpa.json](input/input_3bpa.json)`, with the following exceptions:

- Instead of hidden irreps of `64x0e + 64x0o + 64x1o + 64x1e + 64x2e + 64x2o +  64x3o + 64x3e`, we opt for `128x0e + 128x1o + 128x2e + 128x3o` for simplicity
 with the same feature dimensions.
- We use Kaiming normal, i.e. sampling from $\mathcal{N}(0, \mathrm{std})$ with $\mathrm{std} =
 \sqrt{\frac{4.0}{\mathrm{fan_in}}}$. 
- Isolated atom energies are added to each predicted node energy.

Resulting energy (E) and force (F) RMSE in meV and meV/Å respectively.


| Code     | `[nequip](https://github.com/mir-group/nequip)` | `[mace](https://github.com/ACEsuit/mace)` | `BAM-nequip` (this repo) | `BAM-race` (this repo) |
| -------- | ----------------------------------------------- | ----------------------------------------- | ------------------------ | ---------------------- |
| 300 K E  | 3.3 (0.1)                                       | 3.1 (0.1)                                 | 3.7                      | **3.2**                |
| 300 K F  | 10.8 (0.2)                                      | 11.3 (0.2)                                | 9.6                      | **8.8**                |
| 600 K E  | 11.2 (0.1)                                      | 11.3 (0.3)                                | 11.1                     | **10.1**               |
| 600 K F  | 26.4 (0.1)                                      | 27.3 (0.3)                                | 23.5                     | **22.3**               |
| 1200 K E | 38.5 (1.6)                                      | 40.8 (1.3)                                | 33.5                     | **31.4**               |
| 1200 K F | 76.2 (1.1)                                      | 86.4 (1.5)                                | **67.6**                 | 72.9                   |


### Example: Testing Equivariance

```python
from race_nnx import RACE
from flax import nnx

# Create model
model = RACE(
    n_species=4,
    lmax=3,
    cutoff=5.0,
    hidden_size=128,
    n_layers=3,
    rngs=nnx.Rngs(42),
)

# Run inference
energies, forces = model(graph_data)
```

## Citation

If you use this code, please cite the original NequIP paper:

```bibtex
@article{batzner2022nequip,
  title={E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [e3nn-jax](https://github.com/e3nn/e3nn-jax) for equivariant neural network primitives
- [jraph](https://github.com/deepmind/jraph) for graph neural network utilities
- Original [NequIP](https://github.com/mir-group/nequip) implementation
- [mariogeiger/nequip-jax](https://github.com/mariogeiger/nequip-jax): Another basic implementation of a NequIP style model in JAX.
- [teddykoker/nequip-eqx](https://github.com/teddykoker/nequip-eqx): Better and simple implementation of a NequIP style model in JAX and equinox

