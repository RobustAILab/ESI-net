"""Generate incomplete versions of a multiview dataset for the missing ratios mu used in the paper.

For each view v, floor(mu / V * N) samples are removed (their features are set to 0 and marked in
``missing_matrix``). Every sample keeps one randomly chosen protected view, so no sample loses all
of its views.

Usage:
    python make_incomplete.py --dataset HandWritten --missing_ratios 0.1 0.3 0.5 0.7 0.9

Reads  <data_path>/<prefix>.mat            (complete data, e.g. datasets/handwritten.mat)
Writes <data_path>/<prefix>_missing_<mu>.mat (complete data + missing_matrix), the files read by main.py.
"""
import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat

from presets import canonical_dataset_name, data_file_name


def make_data_incomplete_and_track_missing(X, missing_ratio):
    """
    Process data for each view according to the specified missing ratio, ensuring each instance
    appears in at least one view. Also record and return a matrix indicating which instances
    are missing in which views.

    :param X: list of arrays or dict of arrays, each element is a 2D array or dictionary of 2D arrays.
    :param missing_ratio: specified missing ratio, range between 0 and 1.
    :return: processed X and a boolean matrix indicating missing positions.
    """
    if isinstance(X, dict):
        # X is a dictionary of arrays
        num_views = len(X)
        view_keys = list(X.keys())
        num_instances = X[view_keys[0]].shape[0]
    else:
        # X is a list of arrays
        num_views = len(X)
        num_instances = X[0].shape[0]

    missing_matrix = np.zeros((num_instances, num_views), dtype=bool)
    ratio = missing_ratio / num_views

    # Randomly select a protected view for each instance
    protected_views_for_instances = np.random.randint(num_views, size=num_instances)

    for view_index in range(num_views):
        if isinstance(X, dict):
            view_data = X[view_keys[view_index]]
        else:
            view_data = X[view_index]

        num_missing_instances = int(np.floor(ratio * num_instances))
        possible_indices = [i for i in range(num_instances) if protected_views_for_instances[i] != view_index]

        if len(possible_indices) < num_missing_instances:
            indices_to_remove = possible_indices
        else:
            indices_to_remove = np.random.choice(possible_indices, num_missing_instances, replace=False)

        # Set selected instances as missing in current view
        for idx in indices_to_remove:
            view_data[idx] = 0
            missing_matrix[idx, view_index] = True

        if isinstance(X, dict):
            X[view_keys[view_index]] = view_data
        else:
            X[view_index] = view_data

    return X, missing_matrix


def make_incomplete_file(source_path, output_path, missing_ratio):
    """Create one incomplete copy of ``source_path`` with missing ratio ``missing_ratio``."""
    mat_data = loadmat(source_path, squeeze_me=True)
    Y = mat_data['Y']
    Y = Y.reshape(-1, 1)

    if 'X' in mat_data:
        # Cell-array datasets (HandWritten): X is a 1 x V cell of N x d_v matrices
        X = mat_data['X']
        X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)
        mat_data['X'] = X_incomplete
    else:
        # Datasets stored as separate variables X1, X2, ... (MNIST-USPS, BDGP, Fashion)
        X = {key: mat_data[key] for key in mat_data.keys() if key.startswith('X')}
        X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)
        mat_data.update(X_incomplete)

    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # 1 marks a missing view
    savemat(output_path, mat_data)
    print(f"Processed data and missing matrix saved as: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate incomplete multiview data (<prefix>_missing_<mu>.mat) for ERL-Net.')
    parser.add_argument('--dataset', required=True, type=str,
                        help='MNIST-USPS, BDGP, Fashion or HandWritten (aliases such as "hand" are accepted)')
    parser.add_argument('--missing_ratios', '--mu', dest='missing_ratios', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='Missing ratios mu to generate (default: 0.1 0.3 0.5 0.7 0.9)')
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='Directory containing the complete <prefix>.mat file; outputs are written here')
    parser.add_argument('--source', default=None, type=str,
                        help='Path of the complete .mat file (default: <data_path>/<prefix>.mat)')
    parser.add_argument('--seed', default=None, type=int,
                        help='Optional NumPy seed, applied before each missing ratio')
    args = parser.parse_args()

    dataset = canonical_dataset_name(args.dataset)
    source_path = args.source or os.path.join(args.data_path, data_file_name(dataset))

    for missing_ratio in args.missing_ratios:
        if args.seed is not None:
            np.random.seed(args.seed)
        output_path = os.path.join(args.data_path, data_file_name(dataset, missing_ratio))
        make_incomplete_file(source_path, output_path, missing_ratio)


if __name__ == "__main__":
    main()
