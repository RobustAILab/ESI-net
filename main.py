"""Train or evaluate ERL-Net on an incomplete multiview dataset.

Examples:
    python main.py --dataset HandWritten --missing_ratio 0.5
    python main.py --dataset HandWritten --missing_ratio 0.5 --load_checkpoint --checkpoint checkpoints/hand.pth

Hyperparameters that are not given on the command line are taken from the paper presets in
presets.py (Sec. IV-A3 of the paper).
"""
import os

# Defaults only; values already set in the environment take precedence
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "7")
import torch
import random
import logging
import numpy as np
import argparse
from torch.optim import Adam
from models.erlnet import ERLNet
from utils.training import pretrain, joint_train, valid
from dataprocessing import MultiviewData, get_multiview_data, get_eval_loader
from presets import DATASETS, canonical_dataset_name, get_dataset_seed, get_paper_preset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def default_checkpoint_path(dataset, missing_ratio):
    """Default checkpoint location for a dataset and missing ratio."""
    return os.path.join('./checkpoints', f'{dataset}_missing_{missing_ratio}.pth')


def save_model(model, checkpoint_path):
    """Save model to disk."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(
        model.state_dict(),
        checkpoint_path
    )
    logging.info(f"Checkpoint saved to {checkpoint_path}.")


def build_parser():
    parser = argparse.ArgumentParser(
        description='ERL-Net: energy-guided representation learning for incomplete multiview clustering. '
                    'Options marked [preset] default to the paper value for the chosen dataset (presets.py).')

    # Dataset related
    parser.add_argument('--dataset', default='HandWritten', type=str,
                        help='Dataset: {} (aliases such as "hand" are accepted)'.format(', '.join(DATASETS)))
    parser.add_argument('--missing_ratio', '--mu', dest='missing_ratio', default=0.5, type=float,
                        help='Missing ratio mu; loads <data_path>/<prefix>_missing_<mu>.mat (see README)')
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='[preset] Batch size for training/evaluation (paper: 256)')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed (default: dataset-specific seed from presets.py)')

    # Training related
    parser.add_argument('--learning_rate', default=None, type=float,
                        help='[preset] Learning rate (paper: 1.94e-3)')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--pre_epochs', default=None, type=int,
                        help='[preset] Number of autoencoder pretraining epochs (paper: 200)')
    parser.add_argument('--con_epochs', '--joint_epochs', dest='con_epochs', default=None, type=int,
                        help='[preset] Number of joint training epochs (paper: 300)')

    # Model architecture related
    parser.add_argument('--feature_dim', default=None, type=int,
                        help='[preset] Dimension d_z of the autoencoder embeddings (paper: 64)')
    parser.add_argument('--high_feature_dim', default=None, type=int,
                        help='[preset] Dimension d_r of the common feature space (paper: 20)')
    parser.add_argument('--hidden_dim', default=None, type=int,
                        help='[preset] Hidden dimension of the networks and the GCN (paper: 256)')
    parser.add_argument('--dropout', default=None, type=float,
                        help='[preset] Dropout rate (paper: 0.1)')
    parser.add_argument('--graph_temperature', '--tau_g', dest='graph_temperature', default=None, type=float,
                        help='[preset] tau_g: temperature of the energy-based fusion weights, '
                             'Eqs. (13) and (18) (paper: 0.01); the Eq. (8) energy similarity '
                             'uses a fixed temperature of 0.1')

    # Representation alignment
    parser.add_argument('--temperature', '--tau_c', dest='temperature', default=None, type=float,
                        help='[preset] tau_c: contrastive temperature of L_r-align (paper: 0.5)')
    parser.add_argument('--contrastive_weight', '--beta', dest='contrastive_weight', default=None, type=float,
                        help='[preset] beta: weight of the representation alignment loss L_r-align')
    parser.add_argument('--recon_weight', default=1.0, type=float,
                        help='Weight of the input reconstruction loss L_prec')

    # Imputation branch: mapping loss and energy alignment
    parser.add_argument('--imp_weight', default=1.0, type=float,
                        help='Overall weight of the imputation-branch losses (xi*L_map + alpha*L_ev + w_eg*L_eg)')
    parser.add_argument('--mapping_loss_weight', '--xi', dest='mapping_loss_weight', default=None, type=float,
                        help='[preset] xi: weight of the mapping loss L_map (paper: 1.84)')
    parser.add_argument('--cross_view_energy_weight', '--alpha', dest='cross_view_energy_weight',
                        default=None, type=float,
                        help='[preset] alpha: weight of the energy alignment loss L_ev')
    parser.add_argument('--langevin_weight', default=1.0, type=float,
                        help='w_eg: weight of the Langevin global energy alignment loss L_eg')

    # Langevin dynamics, Eq. (23)
    parser.add_argument('--langevin_steps', '--T', dest='langevin_steps', default=None, type=int,
                        help='[preset] T: number of Langevin steps (paper: 50)')
    parser.add_argument('--langevin_step_size', '--eta', dest='langevin_step_size', default=None, type=float,
                        help='[preset] eta: Langevin step size; the noise is sqrt(eta) * N(0, I) (paper: 0.1)')

    # Energy margin, Eq. (22)
    parser.add_argument('--margin', '--gamma', dest='margin', default=None, type=float,
                        help='[preset] gamma: energy margin of L_eg (paper: 0.1)')

    # Evaluation related
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='If set, load the checkpoint and evaluate without training.')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Checkpoint path to save to / load from '
                             '(default: checkpoints/<dataset>_missing_<mu>.pth)')
    return parser


def resolve_presets(args):
    """Canonicalize the dataset name and fill unset options from the paper presets."""
    args.dataset = canonical_dataset_name(args.dataset)
    for key, value in get_paper_preset(args.dataset).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.seed is None:
        args.seed = get_dataset_seed(args.dataset)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_path(args.dataset, args.missing_ratio)
    return args


def main():
    args = resolve_presets(build_parser().parse_args())
    logging.info('Configuration: {}'.format(vars(args)))

    # Set random seed
    seed = args.seed
    setup_seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: {}'.format(device))

    # Load dataset
    mv_data = MultiviewData(args.dataset, device, path=args.data_path, missing_ratio=args.missing_ratio)
    data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)
    # Evaluation covers every sample in a fixed order (training uses the shuffled loader above)
    eval_loader = get_eval_loader(mv_data, args.batch_size)
    logging.info('Dataset loaded: {} views, {} samples, {} clusters'.format(
        num_views, num_samples, num_clusters))

    # Get dimensions for each view
    view_dims = [mv_data.data_views[i].shape[1] for i in range(num_views)]

    # Initialize model
    model = ERLNet(
        view_dims=view_dims,
        feature_dim=args.feature_dim,
        high_feature_dim=args.high_feature_dim,
        margin=args.margin,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        graph_temperature=args.graph_temperature
    ).to(device)

    # If load_checkpoint is set, load and evaluate without training
    if args.load_checkpoint:
        checkpoint_path = args.checkpoint
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            logging.info("Checkpoint loaded. Evaluating model...")
            acc, nmi, pur = valid(model, device, eval_loader, num_clusters, epoch=0)
            logging.info("Evaluation Results: ACC = {:.4f}, NMI = {:.4f}, PUR = {:.4f}".format(acc, nmi, pur))
            return
        else:
            logging.info(f"No checkpoint found at {checkpoint_path}. Exiting.")
            return

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_acc = 0
    best_nmi = 0
    best_pur = 0
    epoch = 1

    # Stage 1: autoencoder pretraining with L_prec
    while epoch <= args.pre_epochs:
        pretrain(epoch, model, optimizer, data_loader, device)
        epoch += 1

    # Stage 2: joint training with L_total, Eq. (27)
    while epoch <= args.pre_epochs + args.con_epochs:
        avg_losses, _ = joint_train(
            epoch, model, optimizer, data_loader, device,
            temperature=args.temperature,
            mapping_loss_weight=args.mapping_loss_weight,
            cross_view_energy_weight=args.cross_view_energy_weight,
            langevin_weight=args.langevin_weight,
            contrastive_weight=args.contrastive_weight,
            recon_weight=args.recon_weight,
            imp_weight=args.imp_weight,
            langevin_steps=args.langevin_steps,
            langevin_step_size=args.langevin_step_size,
            energy_margin=args.margin
        )

        # Validation
        acc, nmi, pur = valid(model, device, eval_loader, num_clusters, epoch)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_pur = pur
            save_model(model, args.checkpoint)

        epoch += 1

    logging.info('Best Results: ACC = {:.4f}, NMI = {:.4f}, PUR = {:.4f}'.format(
        best_acc, best_nmi, best_pur))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    main()
