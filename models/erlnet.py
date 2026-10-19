"""ERL-Net: energy-guided representation learning network for incomplete multiview clustering.

The model follows Fig. 1 of the paper:
  1) multiview feature extraction (Sec. III-B),
  2) energy-geometric graph encoding (Sec. III-C),
  3) energy-gated imputation (Sec. III-D),
  4) energy-weighted fusion (Sec. III-E).
The alignment losses of Sec. III-E live in ``losses/``.

Note: sub-module attribute names (``encoders``, ``decoders``, ``common_feature``, ``energy_net``,
``global_to_view_mapping``, ``view_graph_fusion``) are kept unchanged so that existing
checkpoints load with ``load_state_dict(strict=True)``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from .networks import EnergyFunction, GlobalToViewMapper
from .energy_graph import EnergyGeometricGraphEncoder


class ERLNet(nn.Module):
    def __init__(self, view_dims, feature_dim=64, high_feature_dim=20,
                 margin=1.0, hidden_dim=256, dropout=0.1, graph_temperature=0.1):
        """
        Args:
            view_dims: Input dimension d_v of every view
            feature_dim: Dimension d_z of the view-specific autoencoder embeddings
            high_feature_dim: Dimension d_r of the common feature space
            margin: Energy margin gamma (stored for reference; the margin is applied in
                    losses.imputation.ImputationLoss)
            hidden_dim: Hidden dimension of the autoencoders, energy function, mappers and GCN
            dropout: Dropout rate
            graph_temperature: Temperature tau_g of the energy-based fusion weights, Eqs. (13)
                               and (18); the Eq. (8) energy similarity uses a fixed 0.1
        """
        super(ERLNet, self).__init__()
        self.view = len(view_dims)
        self.margin = margin
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.view_dims = view_dims
        self._init_networks(view_dims, feature_dim, high_feature_dim, hidden_dim, dropout, graph_temperature)

    def _init_networks(self, view_dims, feature_dim, high_feature_dim, hidden_dim, dropout, graph_temperature):
        # View-specific autoencoders E_v / D_v, Eq. (4)
        self.encoders = nn.ModuleList([
            self._create_encoder(dim, feature_dim, hidden_dim, dropout) for dim in view_dims
        ])
        self.decoders = nn.ModuleList([
            self._create_decoder(dim, feature_dim, hidden_dim, dropout) for dim in view_dims
        ])

        # Shared projection network F into the common space R^{d_r}, Eq. (6)
        self.common_feature = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            nn.LayerNorm(high_feature_dim),
            nn.Dropout(dropout)
        )

        # Shared energy function E_theta (Sec. III-C1)
        self.energy_net = EnergyFunction(
            r_dim=high_feature_dim,
            hidden_dim=hidden_dim
        )

        # Global-to-view mappers Psi_v (Sec. III-D, Eq. (14))
        self.global_to_view_mapping = GlobalToViewMapper(
            global_dim=high_feature_dim,
            view_dims=[high_feature_dim] * self.view,  # All view features have the same dimension
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Energy-geometric graph encoding + energy-weighted fusion (Secs. III-C2 and III-E)
        self.view_graph_fusion = EnergyGeometricGraphEncoder(
            feature_dim=high_feature_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=dropout,
            temperature=graph_temperature
        )

    @staticmethod
    def _create_encoder(input_dim, output_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_decoder(output_dim, input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def _extract_features(self, views):
        """
        Multiview feature extraction (Sec. III-B, Eqs. (4) and (6)) for complete data.

        Args:
            views: List of view data
        """
        xrs, zs, rs = [], [], []

        for v in range(self.view):
            z = self.encoders[v](views[v])
            xr = self.decoders[v](z)
            r = normalize(self.common_feature(z), dim=1)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        return xrs, zs, rs

    def _extract_features_masked(self, views, missing_info=None):
        """
        Multiview feature extraction (Sec. III-B, Eqs. (4) and (6)); outputs of missing views are zeroed.

        Args:
            views: List of view data
            missing_info: Missing information mask [batch_size, n_views], 1 indicates missing, 0 indicates present
                          (i.e. 1 - M in the notation of the paper)
        """
        xrs, zs, rs = [], [], []

        for v in range(self.view):
            if missing_info is not None:
                # 1 indicates missing, 0 indicates present
                mask = (1 - missing_info[:, v]).unsqueeze(1)  # Convert to 1 indicates present, 0 indicates missing

                # Process non-missing data
                x_available = views[v]
                z = self.encoders[v](x_available)
                xr = self.decoders[v](z)
                r = normalize(self.common_feature(z), dim=1)

                # Set missing data to zero (mask=0)
                z = z * mask
                xr = xr * mask
                r = r * mask
            else:
                # Normal processing of non-missing data
                z = self.encoders[v](views[v])
                xr = self.decoders[v](z)
                r = normalize(self.common_feature(z), dim=1)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        return xrs, zs, rs

    def _energy_gated_imputation(self, original_rs, r_hat_features, missing_info):
        """
        Energy-gated imputation (Sec. III-D, Eqs. (16)-(17)).

        Observed features are kept; a missing feature is replaced by its imputed candidate r_hat
        only if E_theta(r_hat) is lower than the average energy e_bar_v of the observed features
        of view v in the current batch (Eq. (7)); otherwise it stays missing.

        Args:
            original_rs: Projected features r [n_views][batch_size, feature_dim] (not GCN-refined)
            r_hat_features: Imputed candidates r_hat = Psi_v(H~) [n_views][batch_size, feature_dim]
            missing_info: Missing information mask [batch_size, n_views], 1 indicates missing, 0 indicates present

        Returns:
            hybrid_features: Valid features r_breve [n_views][batch_size, feature_dim]
            updated_missing_info: Updated missing mask (1 - M*)
            imputation_stats: Imputation statistics
        """
        n_views = self.view
        device = missing_info.device

        # Initialize valid features and updated missing mask
        hybrid_features = []
        updated_missing = missing_info.clone()

        # Statistics
        imputation_details = {
            'view_qualities': {},
            'imputation_rates': {},
            'total_missing': {},
            'actual_imputed': {}
        }

        # Gating thresholds e_bar_v, Eq. (7)
        view_energy_thresholds = self._view_average_energies(original_rs, missing_info)

        for v in range(n_views):
            # Start from the observed features (not GCN-refined)
            hybrid_view_feature = original_rs[v].clone()

            # Find missing samples in current view
            missing_indices = torch.nonzero(missing_info[:, v]).squeeze(1)
            if len(missing_indices.shape) == 0:
                missing_indices = missing_indices.unsqueeze(0)

            # Initialize statistics
            missing_count = len(missing_indices)
            successful_indices = []

            if missing_count > 0:
                # For each missing sample, decide whether to accept the imputed candidate
                for idx in missing_indices:
                    mapped_feature = r_hat_features[v][idx].unsqueeze(0)

                    # Energy of the candidate
                    with torch.no_grad():
                        energy = self.energy_net(mapped_feature).item()

                    # Energy gate, Eq. (16): accept low-energy (reliable) candidates only
                    if energy < view_energy_thresholds[v]:
                        hybrid_view_feature[idx] = mapped_feature.squeeze(0)
                        updated_missing[idx, v] = 0  # Mark as no longer missing (Eq. (17))
                        successful_indices.append(idx.item())

            # Normalize valid features
            hybrid_features.append(normalize(hybrid_view_feature, dim=1))

            # Record statistics
            imputation_details['total_missing'][v] = missing_count
            imputation_details['actual_imputed'][v] = len(successful_indices)
            imputation_details['imputation_rates'][v] = len(
                successful_indices) / missing_count if missing_count > 0 else 0.0

            # Record imputation quality
            if successful_indices:
                idx_tensor = torch.tensor(successful_indices, device=device, dtype=torch.long)
                successful_features = r_hat_features[v][idx_tensor]
                with torch.no_grad():
                    quality = self.energy_net(successful_features).mean().item()
                imputation_details['view_qualities'][v] = quality
            else:
                imputation_details['view_qualities'][v] = float('inf')

        # Calculate overall statistics
        total_missing = sum(imputation_details['total_missing'].values())
        actual_imputed = sum(imputation_details['actual_imputed'].values())
        overall_rate = actual_imputed / total_missing if total_missing > 0 else 0.0

        imputation_stats = {
            'details': imputation_details,
            'overall': {
                'total_missing': total_missing,
                'actual_imputed': actual_imputed,
                'imputation_rate': overall_rate
            }
        }

        return hybrid_features, updated_missing, imputation_stats

    def _view_average_energies(self, original_rs, missing_info):
        """
        Average energy e_bar_v of the observed features of each view in the current batch
        (Eq. (7)), used as the energy-gating threshold.

        Args:
            original_rs: Projected features r [n_views][batch_size, feature_dim] (not GCN-refined)
            missing_info: Missing information mask [batch_size, n_views], 1 indicates missing, 0 indicates present

        Returns:
            view_thresholds: Energy thresholds for each view [n_views]
        """
        n_views = len(original_rs)
        device = original_rs[0].device
        view_thresholds = torch.zeros(n_views, device=device)

        for v in range(n_views):
            # Get mask for non-missing samples (0 indicates present)
            available_mask = (missing_info[:, v] == 0)
            if available_mask.sum() > 0:
                available_indices = torch.nonzero(available_mask).squeeze(1)
                if len(available_indices) > 0:
                    available_features = original_rs[v][available_indices]
                    energies = self.energy_net(available_features).squeeze(-1)
                    view_thresholds[v] = energies.mean()
            else:
                view_thresholds[v] = float('inf')

        return view_thresholds

    def compute_reference_representation(self, rs, missing_mask, temperature=0.1):
        """
        Graph encoding followed by energy-weighted fusion over the available views (Eq. (13)).

        Args:
            rs: Feature list for each view [batch_size, feature_dim]
            missing_mask: Missing information mask [batch_size, n_views], 1 indicates missing, 0 indicates present
            temperature: Unused; the fusion temperature is tau_g (graph_temperature)

        Returns:
            H_global: Fused representation [batch_size, feature_dim]
            energy_weights: Energy-based fusion weights [batch_size, n_views]
        """
        H_global, enhanced_features, adj_matrix, energy_weights = self.view_graph_fusion(
            rs, self.energy_net, missing_mask
        )

        return H_global, energy_weights

    def _reference_representation(self, rs, missing_mask):
        """
        Fused representation computed from the available views (see ``compute_reference_representation``).

        Args:
            rs: Feature list [batch_size, feature_dim]
            missing_mask: Missing information mask [batch_size, n_views], 1 indicates missing, 0 indicates present

        Returns:
            H_global: Fused representation [batch_size, feature_dim]
            energy_weights: Energy-based fusion weights [batch_size, n_views]
        """
        return self.compute_reference_representation(rs, missing_mask)

    def forward(self, views, missing_info=None, mode='pretrain', clustering_info=None):
        """
        Args:
            views: List of view data
            missing_info: Missing information mask, 1 indicates missing, 0 indicates present
            mode: 'pretrain' - feature extraction and graph-encoded fusion only (autoencoder
                               pretraining)
                  'train'    - full ERL-Net forward pass (Algorithm 1), used in joint training and
                               in evaluation (its fourth output is the final H of Eq. (19))
            clustering_info: Unused
        Returns:
            'pretrain': (xrs, zs, rs, H)
            'train':    (xrs, zs, valid_rs, H, updated_missing_info, energy_weights, imputation_stats)
        """
        if mode == 'pretrain':
            xrs, zs, rs = self._extract_features_masked(views, missing_info)
            # Graph-encoded, energy-weighted fusion of the available views
            H, _ = self._reference_representation(rs, missing_info)
            return xrs, zs, rs, H
        else:  # 'train' mode
            # 1. Multiview feature extraction (Sec. III-B)
            if missing_info is not None:
                xrs, zs, rs = self._extract_features_masked(views, missing_info)
            else:
                xrs, zs, rs = self._extract_features(views)
                missing_info = torch.zeros(rs[0].size(0), self.view, device=rs[0].device)

            if missing_info is not None and missing_info.sum() > 0:
                # 2. Energy-geometric graph encoding: r -> GCN -> r~ -> reference representation H~ (Eq. (13))
                enhanced_rs, H_for_mapping, adj_matrix = self.view_graph_fusion.encode(
                    rs, self.energy_net, missing_info
                )

                # 3. Global-to-view mapping: H~ -> imputed candidates r_hat (Eq. (14))
                r_hat_features = self.global_to_view_mapping(H_for_mapping)

                # 4. Energy-gated imputation: observed r + accepted r_hat -> valid features r_breve (Eqs. (16)-(17))
                # Note: the observed (not GCN-refined) features r are used here
                hybrid_features, updated_missing_info, imputation_stats = self._energy_gated_imputation(
                    rs, r_hat_features, missing_info
                )

                # 5. Energy-weighted fusion of the valid features -> final H (Eqs. (18)-(19), no second GCN)
                H_final, energy_weights = self.view_graph_fusion.energy_weighted_fusion(
                    hybrid_features, self.energy_net, updated_missing_info
                )

                return xrs, zs, hybrid_features, H_final, updated_missing_info, energy_weights, imputation_stats

            else:
                # No missing view in this batch: r_breve = r and M* = M, so the final H is the
                # energy-weighted fusion of the observed features (Eqs. (18)-(19))
                H_final, energy_weights = self.view_graph_fusion.energy_weighted_fusion(
                    rs, self.energy_net, missing_info
                )
                return xrs, zs, rs, H_final, missing_info, energy_weights, None
