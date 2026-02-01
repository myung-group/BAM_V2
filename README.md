# BAM-NequIP

A JAX/Flax implementation of NequIP (Neural Equivariant Interatomic Potentials) using the Flax NNX API and e3nn-jax for equivariant operations.

### Overview
To achieve state-of-the-art performance, we created a separate branch to explore multiple code development approaches.

### Training
Trained for 2000 epochs using the following inputs.

1. Input

```json
{
    "cutoff" : 6.0,
    "train_path": "../example-3BPA_300K/dataset/train_300K.xyz",
    "test_path": "../example-3BPA_300K/dataset/test_300K.xyz",
    "valid_frac": 0.1,
    "atomic_numbers": [1, 6, 7, 8],
    "atom_energies" : [-13.587222780835477, -1029.4889999855063, -1484.9814568572233, -2041.9816003861047],
    "hidden_size": 128,
    "lmax": 3,
    "n_layers": 5,
    "radial_basis_size": 8,
    "radial_mlp_size": 64,
    "radial_mlp_layers": 3,
    "radial_polynomial_p": 2.0,
    "mlp_init_scale": 4.0,
    "learning_rate": 0.01,
    "lr_schedule_patience": 20,
    "lr_schedule_factor": 0.8,
    "lr_schedule_min_scale": 0.001,
    "batch_size": 2,
    "n_epochs": 2000,
    "energy_weight": 1.0,
    "force_weight": 1.0,
    "log_every": 100,
    "ema_decay": 0.99,
    "use_scan": false,
    "grad_clip": 1.0,
    "restart": false
}
```

### Comparison of Results

In order to verify correctness of the implementation, we compare performance on
the 3BPA dataset to two different PyTorch NequIP implementations: 

1. The results from Musaelian et al. in ["Learning local equivariant
representations for large-scale atomistic
dynamics"](https://www.nature.com/articles/s41467-023-36329-y), using a version
of their [`nequip`](https://github.com/mir-group/nequip) repository.  
2. The
results from Batatia et al. in ["MACE: Higher Order Equivariant Message
Passing Neural Networks for Fast and Accurate Force
Fields"](https://arxiv.org/abs/2206.07697), using a version of their [`mace`](https://github.com/ACEsuit/mace) repository.

We use the same hyperparameters as [1], which can be viewed in [`configs/3bpa.yml`](configs/3bpa.yml), with the following exceptions:

 * Instead of hidden irreps of `64x0e + 64x0o + 64x1o + 64x1e + 64x2e + 64x2o +
    64x3o + 64x3e`, we opt for `128x0e + 128x1o + 128x2e + 128x3o` for simplicity
    with the same feature dimensions.
 * We use Kaiming normal, i.e. sampling from $\mathcal{N}(0, \mathrm{std})$ with $\mathrm{std} =
    \sqrt{\frac{4.0}{\mathrm{fan\\_in}}}$. 
 * Isolated atom energies are added to each predicted node energy.
    
Resulting energy (E) and force (F) RMSE in meV and meV/Å respectively.

| Code     | `RACE` (race_nnx.py) | `RACE` (race_nnx_4.py) (1100 epoch) | `MACE` (Highest score) | 
|--------------|------|-----------|-----------|
| 300 K E      | 3.52  | *3.48*   | **3.0** |
| 300 K F      | *8.69*  | **8.43**   | 8.8 |
| 600 K E      | 10.19 | **9.32**   | *9.7* |
| 600 K F      | *21.62* | **20.72**  |  21.8  |
| 1200 K E     | 30.30 | *30.12*  |  **29.8**  |
| 1200 K F     | 65.17 | *63.07* |   **62.0** |

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
