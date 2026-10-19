"""Paper hyperparameters of ERL-Net (Sec. IV-A3 of the paper) and dataset names.

``python main.py --dataset <DATASET>`` fills every hyperparameter that is not given on the
command line from these presets. Explicit command-line flags always take precedence.
"""

# Canonical dataset names used in the paper (Table I).
DATASETS = ("MNIST-USPS", "BDGP", "Fashion", "HandWritten")

# Accepted spellings (compared after lower-casing and removing '-', '_' and spaces).
_DATASET_ALIASES = {
    "mnistusps": "MNIST-USPS",
    "bdgp": "BDGP",
    "fashion": "Fashion",
    "handwritten": "HandWritten",
    "hand": "HandWritten",
}

# File-name prefix of each dataset: the complete data is expected at <data_path>/<prefix>.mat and the
# incomplete version with missing ratio mu at <data_path>/<prefix>_missing_<mu>.mat.
DATA_FILE_PREFIX = {
    "MNIST-USPS": "MNIST_USPS",
    "BDGP": "BDGP",
    "Fashion": "Fashion",
    "HandWritten": "handwritten",
}

# Dataset-specific fixed random seeds.
DATASET_SEEDS = {
    "MNIST-USPS": 10,
    "BDGP": 30,
    "Fashion": 10,
    "HandWritten": 42,
}

# Hyperparameters shared by all datasets (CLI dest -> value).
#   learning rate 1.94e-3, batch size 256, 200 pretraining + 300 joint-training epochs,
#   d_z = 64, d_r = 20, hidden dimension 256, dropout 0.1,
#   xi = 1.84, gamma = 0.1, eta = 0.1, T = 50, tau_g = 0.01, tau_c = 0.5.
PAPER_SHARED = {
    "learning_rate": 1.94e-3,
    "batch_size": 256,
    "pre_epochs": 200,
    "con_epochs": 300,
    "feature_dim": 64,
    "high_feature_dim": 20,
    "hidden_dim": 256,
    "dropout": 0.1,
    "mapping_loss_weight": 1.84,   # xi
    "margin": 0.1,                 # gamma
    "langevin_step_size": 0.1,     # eta
    "langevin_steps": 50,          # T
    "graph_temperature": 0.01,     # tau_g (fusion weights, Eqs. (13) and (18))
    "temperature": 0.5,            # tau_c
}

# Dataset-specific loss weights (alpha, beta).
PAPER_DATASET = {
    "MNIST-USPS": {"cross_view_energy_weight": 3.36, "contrastive_weight": 2.94},
    "BDGP": {"cross_view_energy_weight": 3.87, "contrastive_weight": 1.94},
    "Fashion": {"cross_view_energy_weight": 3.39, "contrastive_weight": 2.33},
    "HandWritten": {"cross_view_energy_weight": 3.26, "contrastive_weight": 1.81},
}


def canonical_dataset_name(name):
    """Map a dataset name or alias (e.g. 'hand') to the paper's name (e.g. 'HandWritten')."""
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in _DATASET_ALIASES:
        raise ValueError(
            "Unknown dataset '{}'. Choose one of: {}.".format(name, ", ".join(DATASETS)))
    return _DATASET_ALIASES[key]


def data_file_name(dataset, missing_ratio=None):
    """File name of a dataset: complete if ``missing_ratio`` is None, otherwise the incomplete version."""
    if missing_ratio is None:
        return f"{DATA_FILE_PREFIX[dataset]}.mat"
    return f"{DATA_FILE_PREFIX[dataset]}_missing_{missing_ratio}.mat"


def get_dataset_seed(dataset):
    """Get the dataset-specific random seed."""
    return DATASET_SEEDS.get(dataset, 42)


def get_paper_preset(dataset):
    """Return the full paper preset (CLI dest -> value) for a canonical dataset name."""
    preset = dict(PAPER_SHARED)
    preset.update(PAPER_DATASET[dataset])
    return preset
