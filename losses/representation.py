"""Representation alignment loss L_r-align of ERL-Net (Sec. III-E2, Eqs. (24)-(26))."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyWeightedContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=1.0, device='cuda'):
        """
        Energy-weighted contrastive loss Lambda_v * L_ca(H, R_breve^v) for one view (Eqs. (25)-(26)).

        Args:
            batch_size: Batch size
            temperature: Contrastive temperature tau_c
            device: Device information
        """
        super(EnergyWeightedContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def compute_energy_importance(self, E_H, E_r):
        """
        View importance weight Lambda_v = exp(-|e_bar_g - e_bar*_v|) (Eq. (25)), clipped to [0.1, 2.0]
        and used as a constant (no gradient). Both average energies are taken over the valid
        samples of view v in the current batch.

        Args:
            E_H: Energy of global features [batch_size, 1]
            E_r: Energy of single view features [batch_size, 1]

        Returns:
            importance_weight: View importance weight (scalar)
        """
        # Average energies e_bar_g and e_bar*_v
        avg_E_H = E_H.mean()
        avg_E_r = E_r.mean()

        # Energy gap
        energy_diff = torch.abs(avg_E_H - avg_E_r)

        # A smaller energy gap gives a larger weight
        importance = torch.exp(-energy_diff)

        # Limit weight range to avoid weights that are too large or too small
        importance = torch.clamp(importance, min=0.1, max=2.0)

        return importance.item()

    def forward(self, H, r, model=None, view_idx=None):
        """
        Energy-weighted contrastive loss between H and the valid features of one view.

        Args:
            H: Global representation [batch_size, feature_dim]
            r: Valid features of a single view [batch_size, feature_dim]
            model: ERL-Net model (provides the energy function for Lambda_v)
            view_idx: View index
        """
        # Check if input tensors are valid
        if H is None or r is None or H.size(0) == 0 or r.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        batch_size = r.size(0)
        device = r.device

        # View energy importance weight, default is 1.0
        energy_importance = 1.0

        # Similarity matrix scaled by tau_c
        sim_matrix = torch.matmul(r, H.t()) / self.temperature

        # View importance weight Lambda_v, Eq. (25)
        if model is not None:
            # Calculate energy of H
            E_H = model.energy_net(H)  # [batch_size, 1]

            # Calculate energy of r
            E_r = model.energy_net(r)  # [batch_size, 1]

            # Calculate view importance
            energy_importance = self.compute_energy_importance(E_H, E_r)

        # Extract positive sample similarity (diagonal elements)
        pos_sim = torch.diag(sim_matrix).view(batch_size, 1)

        # Create mask to exclude diagonal (self)
        mask = torch.eye(batch_size, device=device) == 0
        neg_sim = sim_matrix[mask].view(batch_size, -1)

        # Combine positive and negative sample similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # Create labels (positive sample index is 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Contrastive loss L_ca, Eq. (26), in transposed form: anchor r_i, positive H_i, negatives H_j
        base_loss = self.criterion(logits, labels)

        # Apply view importance weight
        loss = energy_importance * base_loss

        return loss


class RepresentationAlignmentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        """
        Representation alignment loss L_r-align = (1/V) * sum_v Lambda_v * L_ca(H, R_breve^v) (Eq. (24)).

        Args:
            batch_size: Batch size
            temperature: Contrastive temperature tau_c
            device: Device information
        """
        super(RepresentationAlignmentLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device

        # Energy-weighted contrastive loss
        self.contrastive_loss = EnergyWeightedContrastiveLoss(
            batch_size=batch_size,
            temperature=temperature,
            device=device
        )

    def forward(self, H, rs, model=None, missing_mask=None):
        """
        Calculate L_r-align over the valid samples U*_v of each view.

        Args:
            H: Global representation [batch_size, feature_dim]
            rs: Valid features of each view, each element has shape [batch_size, feature_dim]
            model: ERL-Net model
            missing_mask: Updated missing mask (1 - M*) [batch_size, n_views], 1 indicates missing, 0 indicates present

        Returns:
            total_loss: Total loss
            loss_details: Details of each loss component
        """
        # Check if input is valid
        if H is None or not rs:
            return torch.tensor(0.0, device=self.device), {}

        batch_size = H.size(0)
        n_views = len(rs)
        device = H.device

        # Initialize total loss and view losses
        total_loss = 0.0
        view_losses = []

        # Calculate contrastive loss for each view
        for v in range(n_views):
            # If missing mask exists, handle non-missing samples
            if missing_mask is not None:
                # Create mask for non-missing samples (0 indicates present)
                non_missing_mask = (missing_mask[:, v] == 0)

                # If all samples in this view are missing, skip
                if non_missing_mask.sum() == 0:
                    view_losses.append(0.0)
                    continue

                # Get non-missing samples
                valid_indices = torch.nonzero(non_missing_mask).squeeze(1)
                valid_H = H[valid_indices]
                valid_r = rs[v][valid_indices]

                # Calculate contrastive loss
                view_loss = self.contrastive_loss(valid_H, valid_r, model, view_idx=v)
            else:
                # Use all samples to calculate contrastive loss
                view_loss = self.contrastive_loss(H, rs[v], model, view_idx=v)

            view_losses.append(view_loss.item())
            total_loss += view_loss

        # Calculate average loss
        if n_views > 0:
            total_loss = total_loss / n_views

        # Loss details
        loss_details = {
            'contrastive_loss': total_loss.item(),
            'view_losses': view_losses
        }

        return total_loss, loss_details
