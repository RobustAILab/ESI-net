"""Training and evaluation loops of ERL-Net (Algorithm 1)."""
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import logging
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from utils.metrics import calculate_clustering_acc, calculate_purity
from losses.imputation import ImputationLoss
from losses.representation import RepresentationAlignmentLoss


class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reconstruction_loss = 0.0
        self.contrastive_loss = 0.0
        self.imputation_loss = 0.0
        self.mapping_loss = 0.0
        self.cross_view_energy_loss = 0.0
        self.langevin_loss = 0.0
        self.total_loss = 0.0
        self.count = 0

    def update(self, recon_loss=0.0, cont_loss=0.0, imp_loss=0.0,
               mapping_loss=0.0, cross_view_energy_loss=0.0, langevin_loss=0.0, total_loss=0.0):
        self.reconstruction_loss += recon_loss
        self.contrastive_loss += cont_loss
        self.imputation_loss += imp_loss
        self.mapping_loss += mapping_loss
        self.cross_view_energy_loss += cross_view_energy_loss
        self.langevin_loss += langevin_loss
        self.total_loss += total_loss
        self.count += 1

    def get_averages(self):
        return {
            'recon': self.reconstruction_loss / self.count,
            'cont': self.contrastive_loss / self.count,
            'imp': self.imputation_loss / self.count,
            'mapping': self.mapping_loss / self.count,
            'cross_view': self.cross_view_energy_loss / self.count,
            'langevin': self.langevin_loss / self.count,
            'total': self.total_loss / self.count
        }


def pretrain(epoch, model, optimizer, data_loader, device):
    """Stage 1 (Algorithm 1, line 4): pretrain the autoencoders with the masked reconstruction loss L_prec, Eq. (5)."""
    model.train()
    tracker = LossTracker()
    criterion = nn.MSELoss(reduction='none')

    pbar = tqdm(data_loader,
                desc=f'Pretrain Epoch {epoch}',
                ncols=100,
                leave=True)

    for batch_idx, (xs, _, missing_info) in enumerate(pbar):
        xs = [x.to(device) for x in xs]
        missing_info = missing_info.to(device)

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs, missing_info, mode='pretrain')

        # Only consider non-missing data when calculating reconstruction loss
        total_recon_loss = 0.0
        total_elements = 0
        for v in range(model.view):
            available_mask = (1 - missing_info[:, v]).unsqueeze(1).float()
            recon_loss_per_element = criterion(xs[v], xrs[v])
            masked_loss = recon_loss_per_element * available_mask
            total_recon_loss += masked_loss.sum()
            total_elements += available_mask.sum()

        if total_elements > 0:
            avg_recon_loss = total_recon_loss / total_elements
        else:
            avg_recon_loss = torch.tensor(0.0, device=device)

        avg_recon_loss.backward()
        optimizer.step()

        tracker.update(
            recon_loss=avg_recon_loss.item(),
            total_loss=avg_recon_loss.item()
        )

        avg_losses = tracker.get_averages()
        pbar.set_postfix({
            'L_prec': f'{avg_losses["recon"]:.4f}',
            'Total': f'{avg_losses["total"]:.4f}'
        })

    avg_losses = tracker.get_averages()
    logging.info(
        'Pretrain Epoch {} - Average Reconstruction Loss: {:.4f}'.format(
            epoch, avg_losses["recon"]
        )
    )


def joint_train(epoch, model, optimizer, data_loader, device,
                temperature=1.0, mapping_loss_weight=1.0,
                cross_view_energy_weight=0.5, langevin_weight=0.5,
                contrastive_weight=1.0, recon_weight=1.0, imp_weight=1.0,
                langevin_steps=10, langevin_step_size=0.1, energy_margin=0.1):
    """
    Stage 2 (Algorithm 1, lines 5-14): joint training of the whole network with
    L_total = L_prec + xi * L_map + alpha * L_ev + w_eg * L_eg + beta * L_r-align (Eq. (27)).

    Args:
        epoch: Current training epoch
        model: ERL-Net model
        optimizer: Optimizer
        data_loader: Data loader
        device: Device
        temperature: Contrastive temperature tau_c
        mapping_loss_weight: Mapping loss weight xi
        cross_view_energy_weight: Energy alignment weight alpha (weight of L_ev)
        langevin_weight: Weight w_eg of the Langevin global energy alignment loss L_eg
        contrastive_weight: Representation alignment weight beta
        recon_weight: Weight of the input reconstruction loss L_prec
        imp_weight: Overall weight of the imputation-branch losses (xi * L_map + alpha * L_ev + w_eg * L_eg)
        langevin_steps: Number T of Langevin steps
        langevin_step_size: Langevin step size eta
        energy_margin: Energy margin gamma of L_eg
    """
    model.train()
    tracker = LossTracker()
    mse = nn.MSELoss(reduction='none')  # Use unreduced MSE to support sample-level weights

    # Representation alignment loss L_r-align
    representation_loss = RepresentationAlignmentLoss(
        batch_size=data_loader.batch_size,
        temperature=temperature,
        device=device
    )

    # Imputation-branch losses: L_map, L_ev and L_eg
    imputation_loss = ImputationLoss(
        mapping_loss_weight=mapping_loss_weight,
        cross_view_energy_weight=cross_view_energy_weight,
        langevin_energy_weight=langevin_weight,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
        energy_margin=energy_margin
    )

    # Track overall imputation statistics
    total_missing = 0
    total_imputed = 0

    pbar = tqdm(data_loader,
                desc=f'Train Epoch {epoch}',
                ncols=150,
                leave=True)

    for batch_idx, (xs, _, missing_info) in enumerate(pbar):
        xs = [x.to(device) for x in xs]
        missing_info = missing_info.to(device)

        optimizer.zero_grad()

        # Forward pass to get features, updated missing mask and imputation statistics
        outputs = model(xs, missing_info, mode='train')
        xrs, zs, imputed_rs, H, updated_missing_info, energy_weights, imputation_stats = outputs

        # Update overall imputation statistics
        if imputation_stats is not None:
            total_missing += imputation_stats['overall']['total_missing']
            total_imputed += imputation_stats['overall']['actual_imputed']
            current_ratio = imputation_stats['overall']['imputation_rate']
        else:
            current_ratio = 0.0

        # 1. Imputation-branch losses (L_map, L_ev, L_eg)
        original_rs = []
        for v in range(model.view):
            r = model.common_feature(zs[v])
            original_rs.append(r)

        imp_loss, imp_details = imputation_loss(
            model=model,
            original_rs=original_rs,
            imputed_rs=imputed_rs,
            H=H,
            original_missing_mask=missing_info,
            updated_missing_mask=updated_missing_info
        )

        # 2. Representation alignment loss L_r-align
        cont_loss, cont_details = representation_loss(
            H=H,
            rs=imputed_rs,
            model=model,
            missing_mask=updated_missing_info
        )

        # 3. Input reconstruction loss L_prec - using original missing_info
        recon_loss = 0.0
        for v in range(model.view):
            valid_mask = (missing_info[:, v] == 0).unsqueeze(1).float()
            view_recon_loss = mse(xs[v], xrs[v])
            masked_loss = view_recon_loss * valid_mask
            total_elements = valid_mask.sum()
            if total_elements > 0:
                view_loss = masked_loss.sum() / total_elements
            else:
                view_loss = torch.tensor(0.0, device=device)
            recon_loss += view_loss

        if model.view > 0:
            recon_loss = recon_loss / model.view

        # 4. Total objective, Eq. (27)
        total_loss = (
                recon_weight * recon_loss +
                contrastive_weight * cont_loss +
                imp_weight * imp_loss
        )

        total_loss.backward()
        optimizer.step()

        # Extract detailed loss components
        mapping_loss = imp_details['mapping_loss'] if 'mapping_loss' in imp_details else 0.0
        cross_view_energy_loss = imp_details[
            'view_energy_alignment_loss'] if 'view_energy_alignment_loss' in imp_details else 0.0
        langevin_loss = imp_details[
            'global_energy_alignment_loss'] if 'global_energy_alignment_loss' in imp_details else 0.0

        # Update loss tracker
        tracker.update(
            recon_loss=recon_loss.item(),
            cont_loss=cont_loss.item(),
            imp_loss=imp_loss.item(),
            mapping_loss=mapping_loss,
            cross_view_energy_loss=cross_view_energy_loss,
            langevin_loss=langevin_loss,
            total_loss=total_loss.item()
        )

        # Update progress bar status
        avg_losses = tracker.get_averages()
        pbar.set_postfix({
            'L_r': f'{avg_losses["cont"]:.4f}',
            'L_prec': f'{avg_losses["recon"]:.4f}',
            'Imp': f'{avg_losses["imp"]:.4f}',
            'L_map': f'{avg_losses["mapping"]:.4f}',
            'L_ev': f'{avg_losses["cross_view"]:.4f}',
            'L_eg': f'{avg_losses["langevin"]:.4f}',
            'Imputed': f'{current_ratio:.2%}'
        })

    # Calculate overall imputation rate
    overall_ratio = total_imputed / total_missing if total_missing > 0 else 0.0

    # Log statistics
    logging.info(f'Epoch {epoch} Overall Imputation Rate: {overall_ratio:.2%} ({total_imputed}/{total_missing})')

    avg_losses = tracker.get_averages()
    logging.info(
        'Epoch {} - Losses: L_r-align={:.4f}, L_prec={:.4f}, Imp={:.4f}, L_map={:.4f}, L_ev={:.4f}, L_eg={:.4f}, Total={:.4f}'.format(
            epoch,
            avg_losses["cont"],
            avg_losses["recon"],
            avg_losses["imp"],
            avg_losses["mapping"],
            avg_losses["cross_view"],
            avg_losses["langevin"],
            avg_losses["total"]
        )
    )

    # Create simplified statistics dictionary
    simplified_stats = {
        'overall': {
            'total_missing': total_missing,
            'actual_imputed': total_imputed,
            'ratio': overall_ratio
        }
    }

    return avg_losses, simplified_stats


def valid(model, device, data_loader, class_num, epoch):
    """Evaluation (Sec. IV): k-means on the final representation H of Eq. (19), i.e. the
    energy-weighted fusion of the observed views and the accepted imputations, computed
    for every sample of data_loader (use dataprocessing.get_eval_loader so that all samples
    are evaluated); reported with ACC, NMI and PUR."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for xs, labels, missing_info in data_loader:
            xs = [x.to(device) for x in xs]
            missing_info = missing_info.to(device)

            # Full ERL-Net forward pass (Algorithm 1) without gradients; its fourth output is the
            # final representation H of Eq. (19)
            H = model(xs, missing_info, mode='train')[3]

            all_features.append(H)
            all_labels.append(torch.as_tensor(labels).clone().detach())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logging.info('Clustering...')
    kmeans = KMeans(n_clusters=class_num, n_init=42)
    pred = kmeans.fit_predict(all_features.cpu().numpy())
    acc = calculate_clustering_acc(all_labels, pred)
    nmi = normalized_mutual_info_score(all_labels.cpu().numpy(), pred)
    pur = calculate_purity(all_labels, pred)

    logging.info('Epoch {} Metrics: ACC={:.4f}, NMI={:.4f}, PUR={:.4f}'.format(
        epoch, acc, nmi, pur))

    return acc, nmi, pur
