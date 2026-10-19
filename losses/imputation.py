"""Losses of the imputation branch of ERL-Net.

* ``mapping_loss``                 -- feature reconstruction loss L_map of the mappers, Eq. (15)
* ``view_energy_alignment_loss``   -- energy alignment across views L_ev, Eq. (21)
* ``global_energy_alignment_loss`` -- global energy alignment L_eg with Langevin-perturbed
                                      samples, Eqs. (22)-(23)

L_e-align = L_ev + L_eg (Eq. (20)). In the total objective (Eq. (27)) L_map is weighted by xi and
L_ev by alpha; L_eg has its own weight (``langevin_energy_weight``).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


class ImputationLoss(nn.Module):
    def __init__(self, mapping_loss_weight=1.0, cross_view_energy_weight=0.5,
                 langevin_energy_weight=0.5, langevin_steps=10, langevin_step_size=0.1,
                 energy_margin=0.1):
        """
        Args:
            mapping_loss_weight: Weight xi of the mapping loss L_map
            cross_view_energy_weight: Weight alpha of the view energy alignment loss L_ev
            langevin_energy_weight: Weight of the global energy alignment loss L_eg
            langevin_steps: Number T of Langevin steps
            langevin_step_size: Langevin step size eta
            energy_margin: Energy margin gamma of L_eg, Eq. (22)
        """
        super(ImputationLoss, self).__init__()
        self.mapping_loss_weight = mapping_loss_weight
        self.cross_view_energy_weight = cross_view_energy_weight
        self.langevin_energy_weight = langevin_energy_weight
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size
        self.energy_margin = energy_margin
        self.mse_loss = nn.MSELoss(reduction='mean')

    def mapping_loss(self, original_rs, model, H, non_missing_mask):
        """
        Mapping loss L_map (Eq. (15)): MSE between the observed view features and the features
        mapped from the global representation by the global-to-view mappers Psi_v.

        Eq. (15) maps the reference representation H~ (Eq. (13)) and compares it with the
        normalized features r. This implementation maps the final representation H (Eq. (19))
        and compares it with the projector output F(z) before L2 normalization (see joint_train
        in utils/training.py). The imputation step itself applies Psi_v to H~.

        Args:
            original_rs: Original view feature list [n_views][batch_size, feature_dim]
            model: ERL-Net model
            H: Global feature representation [batch_size, feature_dim]
            non_missing_mask: Non-missing mask M [batch_size, n_views], 0 indicates missing, 1 indicates present

        Returns:
            loss: Mapping loss
        """
        # Get features mapped from H to each view
        mapped_features = model.global_to_view_mapping(H)

        n_views = len(original_rs)
        device = original_rs[0].device

        total_loss = 0.0
        total_samples = 0

        # Calculate loss for each view
        for v in range(n_views):
            # Get mask for non-missing samples
            view_mask = non_missing_mask[:, v]

            # If there are non-missing samples in this view
            non_missing_count = view_mask.sum().item()
            if non_missing_count > 0:
                # Get indices of non-missing samples
                non_missing_indices = torch.nonzero(view_mask).squeeze(1)

                # Original features and mapped features
                original_features = original_rs[v][non_missing_indices]
                view_mapped_features = mapped_features[v][non_missing_indices]

                # Calculate MSE loss
                view_loss = self.mse_loss(original_features, view_mapped_features)

                total_loss += view_loss * non_missing_count
                total_samples += non_missing_count

        # Return average loss
        if total_samples > 0:
            return total_loss / total_samples
        else:
            return torch.tensor(0.0, device=device)

    def view_energy_alignment_loss(self, imputed_rs, H, updated_missing_mask, energy_net):
        """
        View energy alignment loss L_ev (Eq. (21)): aligns the energy of every valid view
        feature with the energy of the global representation H.
        L_ev = (1 / sum_{i,v} M*_{i,v}) * sum_v sum_i M*_{i,v} * [E_theta(r_breve_i^v) - E_theta(H_i)]^2

        Args:
            imputed_rs: Valid (observed or imputed) feature list [n_views][batch_size, feature_dim]
            H: Global representation after imputation [batch_size, feature_dim]
            updated_missing_mask: Updated missing mask (1 - M*) [batch_size, n_views], 1 indicates missing
            energy_net: Shared energy function E_theta

        Returns:
            loss: View energy alignment loss
        """
        n_views = len(imputed_rs)
        batch_size = imputed_rs[0].size(0)
        device = imputed_rs[0].device

        # Updated indicator M*_{i,v}
        non_missing_mask = 1 - updated_missing_mask  # [batch_size, n_views]

        # E_theta(H_i)
        h_energy = energy_net(H).squeeze(1)  # [batch_size]

        # E_theta(r_breve_i^v) for each view
        view_energies = []
        for v in range(n_views):
            view_energy = energy_net(imputed_rs[v]).squeeze(1)  # [batch_size]
            view_energies.append(view_energy)

        # Stack view energies [batch_size, n_views]
        view_energies = torch.stack(view_energies, dim=1)

        # Expand H energy to [batch_size, n_views]
        h_energy_expanded = h_energy.unsqueeze(1).expand(-1, n_views)

        # [E_theta(r_breve_i^v) - E_theta(H_i)]^2
        energy_diff_squared = (view_energies - h_energy_expanded) ** 2

        # Keep valid views only: M*_{i,v} * [E_theta(r_breve_i^v) - E_theta(H_i)]^2
        masked_loss = energy_diff_squared * non_missing_mask

        # Sum of masked terms and number of valid entries
        total_loss = masked_loss.sum()
        total_non_missing = non_missing_mask.sum()

        # Average over the valid entries
        if total_non_missing > 0:
            return total_loss / total_non_missing
        else:
            return torch.tensor(0.0, device=device)

    def global_energy_alignment_loss(self, H, energy_net):
        """
        Global energy alignment loss L_eg (Eqs. (22)-(23)): contrasts H with its Langevin-perturbed
        version H' with an energy margin gamma,
        L_eg = mean_i ReLU((E_theta(H_i) - E_theta(H'_i) + gamma)^2), as written in Eq. (22).
        H' is obtained with T Langevin steps
        H <- H - (eta / 2) * grad E_theta(H) + sqrt(eta) * N(0, I), each followed by L2 normalization.

        Args:
            H: Global representation [batch_size, feature_dim]
            energy_net: Shared energy function E_theta

        Returns:
            loss: Global energy alignment loss
        """
        batch_size, feature_dim = H.size()
        device = H.device

        # E_theta(H)
        original_energy = energy_net(H).squeeze(1)

        # Langevin chain initialized at H (detached)
        sampled_H = H.clone().detach().requires_grad_(True)

        # T Langevin steps, Eq. (23)
        for step in range(self.langevin_steps):
            # Calculate energy gradient at current point
            energy = energy_net(sampled_H)
            grad_energy = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=sampled_H,
                create_graph=False,
                retain_graph=False
            )[0]

            # Gaussian noise scaled by sqrt(eta)
            noise = torch.randn_like(sampled_H) * math.sqrt(self.langevin_step_size)

            # Langevin update
            sampled_H = (sampled_H
                         - (self.langevin_step_size / 2) * grad_energy
                         + noise)

            # Normalization
            sampled_H = normalize(sampled_H, dim=1)
            sampled_H = sampled_H.detach().requires_grad_(True)

        # E_theta(H')
        sampled_energy = energy_net(sampled_H).squeeze(1)

        # Margin-based energy contrast, Eq. (22): ReLU((E_theta(H) - E_theta(H') + gamma)^2)
        energy_diff =  F.relu(torch.square(original_energy - sampled_energy + self.energy_margin))

        # Return average loss
        return energy_diff.mean()

    def forward(self, model, original_rs, imputed_rs, H, original_missing_mask, updated_missing_mask):
        """
        Weighted sum xi * L_map + alpha * L_ev + w_eg * L_eg.

        Args:
            model: ERL-Net model
            original_rs: Original feature list [n_views][batch_size, feature_dim]
            imputed_rs: Valid (observed or imputed) feature list [n_views][batch_size, feature_dim]
            H: Global representation [batch_size, feature_dim]
            original_missing_mask: Original missing mask (1 - M) [batch_size, n_views], 1 indicates missing
            updated_missing_mask: Updated missing mask (1 - M*) [batch_size, n_views], 1 indicates missing

        Returns:
            total_loss: Total loss
            loss_details: Details of each loss component
        """
        # Convert missing mask to non-missing mask M
        non_missing_mask = 1 - original_missing_mask

        # L_map, Eq. (15)
        mapping_loss = self.mapping_loss(
            original_rs, model, H, non_missing_mask
        )

        # L_ev, Eq. (21)
        cross_view_loss = self.view_energy_alignment_loss(
            imputed_rs, H, updated_missing_mask, model.energy_net
        )

        # L_eg, Eqs. (22)-(23)
        langevin_loss = self.global_energy_alignment_loss(H, model.energy_net)

        # Calculate total loss
        total_loss = (
                self.mapping_loss_weight * mapping_loss +
                self.cross_view_energy_weight * cross_view_loss +
                self.langevin_energy_weight * langevin_loss
        )

        # Loss details
        loss_details = {
            'mapping_loss': mapping_loss.item(),
            'view_energy_alignment_loss': cross_view_loss.item(),
            'global_energy_alignment_loss': langevin_loss.item(),
            'total_imputation_loss': total_loss.item()
        }

        return total_loss, loss_details
