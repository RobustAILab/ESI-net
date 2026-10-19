# Not All, but the Right Ones: Energy-Guided Representation Learning for Incomplete Multiview Clustering

This is the official PyTorch implementation of **ERL-Net** (energy-guided representation learning network).

**Ziyu Wang, Jian Li, Yiming Du, Rui Ning, Shuai Hao, Daniel Takabi, Lusi Li**
Old Dominion University

*IEEE Transactions on Neural Networks and Learning Systems*, 2026.
DOI: [10.1109/TNNLS.2026.3702954](https://doi.org/10.1109/TNNLS.2026.3702954)

## Overview

Incomplete multiview clustering (IMVC) aims to find shared cluster structure in data where some views are missing. ERL-Net is a selective imputation framework. It does not try to recover every missing view. Instead, a learned energy function decides which imputed views are reliable enough to keep, and the same energy scores guide fusion and alignment. ERL-Net has four components:

1. **Multiview feature extraction** (Sec. III-B). View-specific autoencoders learn embeddings `z^v`. A shared projection network `F` maps them into a common feature space of dimension `d_r = 20`.
2. **Energy-geometric graph encoding** (Sec. III-C). A shared energy function `E_θ` (an MLP with a SoftPlus output) scores how reliable each feature is. For each sample, an instance-specific view graph is built from the product of an energy similarity and a geometric (cosine) similarity. A two-layer view-shared GCN propagates information over this graph. An energy-weighted fusion of the refined features then gives a reference representation `H̃`.
3. **Energy-gated imputation** (Sec. III-D). Global-to-view mappers `Ψ_v` map `H̃` to a candidate feature for each missing view. A candidate is kept only if its energy is below the average energy of the observed features of that view. Otherwise, the view stays missing.
4. **Energy-weighted fusion and alignment** (Sec. III-E). Observed features and accepted imputations are fused into the final representation `H`, with weights based on energy. Two losses keep the views consistent:
   - an energy alignment loss `L_e-align`, made of a cross-view term `L_ev` and a global term `L_eg` that compares `H` with Langevin-perturbed samples using a margin `γ`;
   - an energy-weighted contrastive representation alignment loss `L_r-align` with temperature `τ_c`.

Training has two stages (Algorithm 1). First, the autoencoders are pretrained with the masked reconstruction loss `L_prec`. Then the whole network is trained jointly with `L_total` (Eq. (27)). Cluster labels are obtained by running k-means on the final representation `H` of Eq. (19) of all samples.

### Where each part of the paper is in the code

| Paper | Code |
| --- | --- |
| Autoencoders `E_v`, `D_v` and projection `F` (Eqs. (4)-(6)) | `ERLNet.encoders`, `ERLNet.decoders`, `ERLNet.common_feature` in `models/erlnet.py` |
| Shared energy function `E_θ` (Sec. III-C1) | `EnergyFunction` in `models/networks.py` |
| Energy-geometric view graph `A_i` (Eqs. (8)-(10)); see the notes under [Hyperparameters](#hyperparameters) for the temperature of Eq. (8) | `EnergyGeometricGraphConstructor` in `models/energy_graph.py` |
| View-shared GCN (Eqs. (11)-(12)) | `ViewGCN`, `GraphConvLayer` in `models/energy_graph.py` |
| Reference representation `H̃` (Eq. (13)) | `EnergyGeometricGraphEncoder.encode` in `models/energy_graph.py` |
| Global-to-view mappers `Ψ_v` (Eq. (14)) | `GlobalToViewMapper` in `models/networks.py` |
| Energy gate and updated indicator `M*` (Eqs. (16)-(17)) | `ERLNet._energy_gated_imputation` in `models/erlnet.py` |
| Energy-weighted fusion into `H` (Eqs. (18)-(19)) | `EnergyGeometricGraphEncoder.energy_weighted_fusion` in `models/energy_graph.py` |
| Input reconstruction loss `L_prec` (Eq. (5)) | `pretrain` and `joint_train` in `utils/training.py` |
| Mapping loss `L_map` (Eq. (15)) | `ImputationLoss.mapping_loss` in `losses/imputation.py` |
| Energy alignment `L_ev` (Eq. (21)) | `ImputationLoss.view_energy_alignment_loss` in `losses/imputation.py` |
| Energy alignment `L_eg` with Langevin dynamics (Eqs. (22)-(23)) | `ImputationLoss.global_energy_alignment_loss` in `losses/imputation.py` |
| Representation alignment `L_r-align` (Eqs. (24)-(26)) | `RepresentationAlignmentLoss`, `EnergyWeightedContrastiveLoss` in `losses/representation.py` |
| Training procedure (Algorithm 1) | `main.py`, `utils/training.py` |

## Repository layout

```
.
├── main.py                 # training / evaluation entry point
├── presets.py              # paper hyperparameters, dataset names and seeds
├── dataprocessing.py       # data loading for the four benchmarks
├── make_incomplete.py      # generates incomplete data for a given missing ratio mu
├── models/
│   ├── erlnet.py           # ERLNet: feature extraction, energy-gated imputation, forward pass
│   ├── energy_graph.py     # energy-geometric graph encoding and energy-weighted fusion
│   └── networks.py         # energy function E_theta and global-to-view mappers Psi_v
├── losses/
│   ├── imputation.py       # L_map, L_ev, L_eg (Langevin dynamics)
│   └── representation.py   # L_r-align (energy-weighted contrastive loss)
├── utils/
│   ├── training.py         # pretraining, joint training and evaluation loops
│   └── metrics.py          # ACC and PUR
├── datasets/
│   └── handwritten_missing_0.5.mat   # example data: HandWritten, mu = 0.5
└── checkpoints/
    └── hand.pth                      # example checkpoint for HandWritten
```

## Installation

```bash
git clone https://github.com/sunway677/ERL-Net.git
cd ERL-Net
```

The pinned versions in `requirements.txt` need Python 3.10-3.12; `environment.yml` creates a Python 3.10 environment. A CUDA-capable GPU is recommended. The code also runs on the CPU, but slowly. GPUs that need CUDA 12.8 or later (e.g., NVIDIA Blackwell) are not supported by the pinned PyTorch 2.5.1 build; on such GPUs, install a newer PyTorch build (the code has also been run with PyTorch 2.11 + CUDA 12.8).

```bash
conda env create -f environment.yml
conda activate erlnet
```

Or install the packages with pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the reference package versions. Install the PyTorch build that matches your CUDA version by following the instructions at https://pytorch.org.

## Datasets

The paper uses four benchmarks: MNIST-USPS, BDGP, Fashion (Fashion-MNIST) and HandWritten (UCI multiple features). Table I of the paper gives their statistics. The datasets are not included in this repository, except for the incomplete HandWritten example `datasets/handwritten_missing_0.5.mat` (the complete `handwritten.mat` is not included).

Put the complete `.mat` files in `datasets/` under these names:

| Dataset | `--dataset` | Views | Complete file | Variables |
| --- | --- | --- | --- | --- |
| MNIST-USPS | `MNIST-USPS` | 2 | `MNIST_USPS.mat` | `X1`, `X2`, `Y` |
| BDGP | `BDGP` | 2 | `BDGP.mat` | `X1`, `X2`, `Y` |
| Fashion | `Fashion` | 3 | `Fashion.mat` | `X1`, `X2`, `X3` (each `N x 28 x 28`), `Y` |
| HandWritten | `HandWritten` (or `hand`) | 6 | `handwritten.mat` | `X` (`1 x 6` cell of `N x d_v` matrices), `Y` |

Dataset names are not case-sensitive, and `-`/`_` are ignored. For example, `mnist_usps` and `hand` are accepted.

### Generating incomplete data

`main.py` reads the incomplete version `datasets/<prefix>_missing_<mu>.mat`, for example `datasets/MNIST_USPS_missing_0.5.mat`. To create these files for the missing ratios used in the paper, run:

```bash
python make_incomplete.py --dataset MNIST-USPS  --missing_ratios 0.1 0.3 0.5 0.7 0.9
python make_incomplete.py --dataset BDGP        --missing_ratios 0.1 0.3 0.5 0.7 0.9
python make_incomplete.py --dataset Fashion     --missing_ratios 0.1 0.3 0.5 0.7 0.9
python make_incomplete.py --dataset HandWritten --missing_ratios 0.1 0.3 0.5 0.7 0.9
```

For each view, `floor(mu * N / V)` samples are removed: their features are set to zero and they are marked in `missing_matrix`. `missing_matrix` is an `N x V` matrix in which `1` marks a missing view. Each sample keeps one randomly chosen view, so no sample loses all of its views. The missing patterns are random. Pass `--seed` to make them reproducible.

The repository includes `datasets/handwritten_missing_0.5.mat`, the HandWritten dataset (2000 samples, 6 views, 10 classes) with `mu = 0.5`. The bundled checkpoint `checkpoints/hand.pth` was trained on this missing pattern. Running the HandWritten command above with `0.5` overwrites this file with a new random pattern, so leave out `0.5` (e.g. `--missing_ratios 0.1 0.3 0.7 0.9`) to keep it.

## Training

`main.py` takes every hyperparameter it is not given from the paper presets in `presets.py`, so one flag is enough to choose the dataset:

```bash
python main.py --dataset MNIST-USPS  --missing_ratio 0.5
python main.py --dataset BDGP        --missing_ratio 0.5
python main.py --dataset Fashion     --missing_ratio 0.5
python main.py --dataset HandWritten --missing_ratio 0.5
```

`--missing_ratio` (alias `--mu`) selects the data file. Each run has two stages:

1. 200 epochs of autoencoder pretraining.
2. 300 epochs of joint training. After every joint-training epoch, k-means is run on the final representation `H` (Eq. (19)) of all samples, and ACC, NMI and PUR are logged.

The checkpoint with the best ACC is saved to `checkpoints/<dataset>_missing_<mu>.pth`. Use `--checkpoint` to choose a different path.

Any preset can be overridden on the command line, using either the original flag or its paper-symbol alias:

```bash
python main.py --dataset Fashion --missing_ratio 0.7 --alpha 3.0 --beta 2.0 --tau_g 0.05
```

Run `python main.py --help` to see all options.

## Evaluation with the bundled checkpoint

```bash
python main.py --dataset HandWritten --missing_ratio 0.5 --load_checkpoint --checkpoint checkpoints/hand.pth
```

This command loads the weights, runs k-means on the final representation `H` (Eq. (19)) of all 2,000 samples and prints ACC, NMI and PUR.

## Hyperparameters

`python main.py --dataset <DATASET>` uses the values from Sec. IV-A of the paper:

| Symbol | Meaning | Flag (alias) | MNIST-USPS | BDGP | Fashion | HandWritten |
| --- | --- | --- | --- | --- | --- | --- |
| `α` | energy alignment weight (`L_ev`) | `--cross_view_energy_weight` (`--alpha`) | 3.36 | 3.87 | 3.39 | 3.26 |
| `β` | representation alignment weight (`L_r-align`) | `--contrastive_weight` (`--beta`) | 2.94 | 1.94 | 2.33 | 1.81 |
| `ξ` | mapping loss weight (`L_map`) | `--mapping_loss_weight` (`--xi`) | 1.84 | 1.84 | 1.84 | 1.84 |
| `γ` | energy margin in `L_eg` | `--margin` (`--gamma`) | 0.1 | 0.1 | 0.1 | 0.1 |
| `η` | Langevin step size | `--langevin_step_size` (`--eta`) | 0.1 | 0.1 | 0.1 | 0.1 |
| `T` | Langevin steps | `--langevin_steps` (`--T`) | 50 | 50 | 50 | 50 |
| `τ_g` | temperature of the energy-based fusion weights, Eqs. (13) and (18) (see notes) | `--graph_temperature` (`--tau_g`) | 0.01 | 0.01 | 0.01 | 0.01 |
| `τ_c` | contrastive temperature | `--temperature` (`--tau_c`) | 0.5 | 0.5 | 0.5 | 0.5 |
| `μ` | missing ratio | `--missing_ratio` (`--mu`) | 0.1-0.9 | 0.1-0.9 | 0.1-0.9 | 0.1-0.9 |
| | learning rate (Adam) | `--learning_rate` | 1.94e-3 | 1.94e-3 | 1.94e-3 | 1.94e-3 |
| | batch size | `--batch_size` | 256 | 256 | 256 | 256 |
| | pretraining / joint-training epochs | `--pre_epochs` / `--con_epochs` | 200 / 300 | 200 / 300 | 200 / 300 | 200 / 300 |
| `d_z` / `d_r` | embedding / common-space dimension | `--feature_dim` / `--high_feature_dim` | 64 / 20 | 64 / 20 | 64 / 20 | 64 / 20 |
| | hidden dimension (networks and GCN) / dropout | `--hidden_dim` / `--dropout` | 256 / 0.1 | 256 / 0.1 | 256 / 0.1 | 256 / 0.1 |
| | random seed | `--seed` | 10 | 30 | 10 | 42 |

Notes:
- In the paper, `τ_g` is the graph construction temperature of Eq. (8) and is also used in Eqs. (13) and (18). In this implementation, the energy similarity of Eq. (8) uses a fixed temperature of 0.1, and `--tau_g` (0.01) sets only the temperature of the fusion weights in Eqs. (13) and (18).
- The Langevin update follows Eq. (23): the noise term is `sqrt(η) · ε_t` with `ε_t ~ N(0, I)`, so its scale is set by `η` alone. The noise scale `ε = 0.01` listed in Sec. IV-A of the paper has no counterpart in the code, and there is no flag for it.
- In Eq. (27), `α` multiplies the whole energy alignment loss `L_e-align = L_ev + L_eg`. In this implementation `α` (`--alpha`) weights `L_ev`, and `L_eg` has its own weight `--langevin_weight` (default 1).
- Other weights are set to 1 by default: `--recon_weight` (weight of `L_prec`) and `--imp_weight` (overall weight of the imputation-branch losses).
- Sec. IV-A3 of the paper describes the encoder as a three-layer network with structure `d_v-256-64`. In this implementation each encoder has three linear layers, `d_v-256-256-64` (ReLU and dropout after the first two), and each decoder mirrors it (`64-256-256-d_v`). Both hidden layers have width `--hidden_dim`.
- The gating threshold `ē_v` of Eq. (7) is the mean energy of the observed features of view `v` in the current mini-batch, not over the whole dataset. This holds in training and in evaluation, so the accepted imputations, `H` and the reported metrics depend on `--batch_size` (paper: 256).
- `ViewGCN` differs from Eq. (12) in a few details: the graph-convolution layers have a bias term; ReLU and dropout are applied after every layer except the last; the GCN output is added to the input features `R_i` (residual connection) and passed through LayerNorm; and each refined view feature is L2-normalized.
- Eq. (15) trains `Ψ_v` on the reference representation `H̃` (Eq. (13)) against the normalized features `r`. In this implementation `L_map` maps the final representation `H` (Eq. (19)) and compares it with the projector output `F(z)` before L2 normalization. The imputation step itself applies `Ψ_v` to `H̃`, as in Eq. (14).
- In Eq. (26) the anchor is `H_i` and the negatives are `r̆_j^v`. The implementation uses the transposed form: it anchors on `r̆_i^v` and contrasts it against `H_j` (row-wise cross-entropy over `r̆^v Hᵀ / τ_c`, with positives on the diagonal). The view weight `Λ_v` of Eq. (25) is computed per mini-batch over the valid samples of view `v`, is treated as a constant (no gradient) and is clipped below at 0.1.

## Metrics

Clustering performance is reported with three metrics. Higher is better for all of them.

- **ACC**: clustering accuracy with the best one-to-one matching between clusters and classes (Hungarian algorithm).
- **NMI**: normalized mutual information.
- **PUR**: purity.

ACC and PUR are implemented in `utils/metrics.py`. NMI comes from scikit-learn.

## Reproducibility

Each dataset uses a fixed seed (see the table above). Results can still vary slightly between runs, GPUs and library versions, and with different random missing patterns. They may therefore not match every number in the paper exactly.

## Citation

If you find this code useful, please cite:

```bibtex
@article{wang2026erlnet,
  title={Not All, but the Right Ones: Energy-Guided Representation Learning for Incomplete Multiview Clustering},
  author={Wang, Ziyu and Li, Jian and Du, Yiming and Ning, Rui and Hao, Shuai and Takabi, Daniel and Li, Lusi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026},
  doi={10.1109/TNNLS.2026.3702954}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
