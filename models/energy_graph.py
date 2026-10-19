"""Energy-geometric graph encoding (Sec. III-C2) and energy-weighted fusion (Sec. III-E).

* ``EnergyGeometricGraphConstructor`` -- instance-specific view graphs A_i, Eqs. (8)-(10).
* ``GraphConvLayer`` / ``ViewGCN``    -- view-shared GCN over A_i, Eqs. (11)-(12).
* ``EnergyGeometricGraphEncoder``     -- graph encoding followed by energy-weighted
                                         fusion: reference representation H~ (Eq. (13))
                                         and final representation H (Eqs. (18)-(19)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


class EnergyGeometricGraphConstructor(nn.Module):
    """Instance-specific view graph from joint energy-geometric similarity (Eqs. (8)-(10))."""

    def __init__(self, fusion_type='product', temperature=0.1,
                 energy_weight=0.5, cosine_weight=0.5):
        """
        Args:
            fusion_type: How the two similarities are combined, 'product' (Eq. (10), used by
                         ERL-Net) or 'weighted_sum'
            temperature: Temperature of the energy similarity in Eq. (8). ViewGCN builds this
                         constructor with the default value 0.1, which is fixed: --tau_g does
                         not change it (in the paper, tau_g is also used in Eq. (8))
            energy_weight: Weight of the energy similarity (only for 'weighted_sum')
            cosine_weight: Weight of the geometric similarity (only for 'weighted_sum')
        """
        super(EnergyGeometricGraphConstructor, self).__init__()
        self.fusion_type = fusion_type
        self.temperature = temperature
        self.energy_weight = energy_weight
        self.cosine_weight = cosine_weight

    def forward(self, view_features, energy_net, missing_mask=None):
        """
        Build the row-normalized adjacency matrices A_i over the observed views of each sample.

        Args:
            view_features: List of view features [n_views][batch_size, feature_dim]
            energy_net: Shared energy function E_theta
            missing_mask: Missing mask [batch_size, n_views], 1 indicates missing

        Returns:
            adj_matrices: Row-normalized adjacency matrices [batch_size, n_views, n_views]
        """
        batch_size = view_features[0].size(0)
        n_views = len(view_features)
        device = view_features[0].device

        # Energy E_theta(r_i^v) of each view feature
        view_energies = []
        for v in range(n_views):
            energy = energy_net(view_features[v]).squeeze(-1)  # [batch_size]
            view_energies.append(energy)

        # Convert energies to tensor [batch_size, n_views]
        energy_tensor = torch.stack(view_energies, dim=1)

        # Initialize adjacency matrices
        adj_matrices = torch.zeros(batch_size, n_views, n_views, device=device)

        # Pairwise similarity between views v != t (no self-loops), for every sample i
        for i in range(n_views):
            for j in range(n_views):
                if i != j:  # Exclude self-loops
                    # 1. Energy similarity, Eq. (8)
                    energy_diff = torch.abs(energy_tensor[:, i].unsqueeze(1) - energy_tensor[:, j].unsqueeze(1))
                    energy_similarity = torch.exp(-energy_diff / self.temperature)

                    # 2. Geometric (cosine) similarity, Eq. (9)
                    features_i = view_features[i]
                    features_j = view_features[j]

                    # Batch cosine similarity [batch_size]
                    cosine_similarity = F.cosine_similarity(features_i, features_j, dim=1)

                    # Rescale cosine similarity from [-1, 1] to [0, 1]
                    cosine_similarity = (cosine_similarity + 1) / 2

                    # 3. Combine the two similarities
                    if self.fusion_type == 'weighted_sum':
                        # Weighted sum
                        combined_similarity = (
                                self.energy_weight * energy_similarity.squeeze(1) +
                                self.cosine_weight * cosine_similarity
                        )
                    else:  # 'product'
                        # Element-wise product, Eq. (10): an edge is kept only if both similarities are high
                        combined_similarity = energy_similarity.squeeze(1) * cosine_similarity

                    # Update adjacency matrix
                    adj_matrices[:, i, j] = combined_similarity

        # Restrict the graph to the observed views O_i
        if missing_mask is not None:
            # Expand missing mask [batch_size, n_views, 1]
            expanded_mask = missing_mask.unsqueeze(2).float()
            # Create row-level mask and column-level mask
            row_mask = expanded_mask.expand(-1, -1, n_views)
            col_mask = expanded_mask.transpose(1, 2).expand(-1, n_views, -1)

            # Set rows and columns corresponding to missing views to 0
            masked_adj = adj_matrices * (1 - row_mask) * (1 - col_mask)

            # Row normalization (A_hat_i in Eq. (12))
            row_sums = masked_adj.sum(dim=2, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-10)  # Prevent division by zero
            normalized_adj = masked_adj / row_sums

            return normalized_adj
        else:
            # Row normalization (A_hat_i in Eq. (12))
            row_sums = adj_matrices.sum(dim=2, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-10)  # Prevent division by zero
            normalized_adj = adj_matrices / row_sums

            return normalized_adj


class GraphConvLayer(nn.Module):
    """Graph convolution layer (one step of Eq. (12) before the activation)."""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features, adj_matrix):
        """
        Perform graph convolution operation

        Args:
            features: Input features [batch_size, n_views, in_features]
            adj_matrix: Adjacency matrix [batch_size, n_views, n_views]

        Returns:
            output: Output features [batch_size, n_views, out_features]
        """
        # Message passing: feature aggregation
        support = torch.bmm(adj_matrix, features)  # [batch_size, n_views, in_features]

        # Feature transformation
        output = self.linear(support)  # [batch_size, n_views, out_features]

        return output


class ViewGCN(nn.Module):
    """View-shared GCN over the instance-specific view graphs (Eqs. (11)-(12)).

    The view graphs are built by an EnergyGeometricGraphConstructor with its default
    temperature, so the energy similarity of Eq. (8) uses a fixed temperature of 0.1.
    tau_g (--tau_g) only sets the fusion-weight temperature of Eqs. (13) and (18) in
    EnergyGeometricGraphEncoder.

    Compared with Eq. (12): the graph-convolution layers have a bias term; ReLU and dropout
    are applied after every layer except the last; the GCN output is added to the input
    features R_i (residual connection) and passed through LayerNorm; and each refined view
    feature is L2-normalized.
    """

    def __init__(self, feature_dim, hidden_dim, n_layers=2, dropout=0.1):
        """
        Args:
            feature_dim: Feature dimension (d_r)
            hidden_dim: Hidden layer dimension
            n_layers: Number of GCN layers
            dropout: Dropout ratio
        """
        super(ViewGCN, self).__init__()

        # Eq. (8) energy similarity with the fixed default temperature 0.1 (not tau_g)
        self.graph_constructor = EnergyGeometricGraphConstructor()

        # Multi-layer GCN
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(GraphConvLayer(feature_dim, hidden_dim))

        # Intermediate layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        # Last layer
        if n_layers > 1:
            self.layers.append(GraphConvLayer(hidden_dim, feature_dim))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, view_features, energy_net, missing_mask=None):
        """
        Refine the view features by message passing over A_i.

        Args:
            view_features: List of view features [n_views][batch_size, feature_dim]
            energy_net: Shared energy function E_theta
            missing_mask: Missing mask [batch_size, n_views], 1 indicates missing

        Returns:
            enhanced_features: GCN-refined view features r~ [n_views][batch_size, feature_dim]
            adj_matrix: View graph adjacency matrix [batch_size, n_views, n_views]
        """
        batch_size = view_features[0].size(0)
        n_views = len(view_features)
        feature_dim = view_features[0].size(1)
        device = view_features[0].device

        # Stacked feature matrix R_i [batch_size, n_views, feature_dim], Eq. (11)
        stacked_features = torch.stack(view_features, dim=1)

        # Instance-specific view graphs, Eqs. (8)-(10)
        adj_matrix = self.graph_constructor(view_features, energy_net, missing_mask)

        # GCN propagation, Eq. (12)
        x = stacked_features
        for i, layer in enumerate(self.layers):
            # Graph convolution
            x = layer(x, adj_matrix)

            # Apply activation function and dropout except for the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Residual connection + Layer Normalization
        x = x + stacked_features
        x = self.layer_norm(x)

        # Split matrix back to list [n_views][batch_size, feature_dim]
        enhanced_features = []
        for v in range(n_views):
            view_feature = x[:, v, :]
            enhanced_features.append(normalize(view_feature, dim=1))

        return enhanced_features, adj_matrix


class EnergyGeometricGraphEncoder(nn.Module):
    """Energy-geometric graph encoding (Sec. III-C2) with energy-weighted fusion (Eqs. (13), (18)-(19))."""

    def __init__(self, feature_dim, hidden_dim=128, n_layers=2, dropout=0.1, temperature=0.1):
        """
        Args:
            feature_dim: Feature dimension (d_r)
            hidden_dim: Hidden layer dimension of the GCN
            n_layers: Number of GCN layers
            dropout: Dropout ratio
            temperature: Temperature tau_g of the energy-based fusion weights, Eqs. (13) and (18).
                         It is not used for the graph construction of Eq. (8), whose
                         temperature is fixed at 0.1 (see ViewGCN)
        """
        super(EnergyGeometricGraphEncoder, self).__init__()

        self.view_gcn = ViewGCN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        self.temperature = temperature

    def encode(self, rs, energy_net, missing_mask=None):
        """
        Graph encoding and reference representation (Sec. III-C2, Eq. (13)).

        Args:
            rs: Projected view features r [n_views][batch_size, feature_dim]
            energy_net: Shared energy function E_theta
            missing_mask: Missing mask [batch_size, n_views], 1 indicates missing

        Returns:
            enhanced_rs: GCN-refined view features r~ [n_views][batch_size, feature_dim]
            H_for_mapping: Reference representation H~ [batch_size, feature_dim]
            adj_matrix: Adjacency matrices A_i
        """
        # 1. GCN refinement
        enhanced_features, adj_matrix = self.view_gcn(rs, energy_net, missing_mask)

        # 2. Reference representation H~ from the refined features, Eq. (13)
        H_for_mapping = self._fuse_reference(enhanced_features, energy_net, missing_mask)

        return enhanced_features, H_for_mapping, adj_matrix

    def energy_weighted_fusion(self, hybrid_features, energy_net, missing_mask=None):
        """
        Energy-weighted fusion of the valid features (Sec. III-E, Eqs. (18)-(19)).

        Args:
            hybrid_features: Valid features r_breve (observed or accepted imputations)
                             [n_views][batch_size, feature_dim]
            energy_net: Shared energy function E_theta
            missing_mask: Updated missing mask (1 - M*) [batch_size, n_views], 1 indicates missing

        Returns:
            H_final: Final global representation H [batch_size, feature_dim]
            energy_weights: Fusion weights w* [batch_size, n_views]
        """
        # Energy-based fusion weights, Eq. (18)
        energy_weights = self._energy_weights(hybrid_features, energy_net, missing_mask)

        # Weighted fusion, Eq. (19)
        stacked_features = torch.stack(hybrid_features, dim=1)
        weights_expanded = energy_weights.unsqueeze(2)
        H_final = torch.sum(weights_expanded * stacked_features, dim=1)

        # Normalization
        H_final = normalize(H_final, dim=1)

        return H_final, energy_weights

    def _fuse_reference(self, features, energy_net, missing_mask=None):
        """
        Energy-weighted fusion of the GCN-refined features into H~ (Eq. (13)).
        """
        energy_weights = self._energy_weights(features, energy_net, missing_mask)

        # Weighted fusion
        stacked_features = torch.stack(features, dim=1)
        weights_expanded = energy_weights.unsqueeze(2)
        H = torch.sum(weights_expanded * stacked_features, dim=1)

        # Normalization
        H = normalize(H, dim=1)

        return H

    def _energy_weights(self, features, energy_net, missing_mask=None):
        """
        Energy-based fusion weights over the available views (Eqs. (13) and (18)):
        w_{i,v} proportional to exp(-E_theta(r_i^v) / tau_g), masked to the available views and
        normalized per sample. The per-sample minimum energy is subtracted for numerical stability.
        """
        batch_size = features[0].size(0)
        n_views = len(features)
        device = features[0].device

        # Calculate energy for each view feature
        view_energies = torch.zeros(batch_size, n_views, device=device)

        for v in range(n_views):
            # Handle non-missing samples
            if missing_mask is not None:
                non_missing_mask = (missing_mask[:, v] == 0)
                if non_missing_mask.sum() > 0:
                    valid_indices = torch.nonzero(non_missing_mask).squeeze(1)
                    if len(valid_indices) > 0:
                        valid_features = features[v][valid_indices]
                        energies = energy_net(valid_features).squeeze(-1)
                        view_energies[valid_indices, v] = energies

                    # Assign large energy values to missing samples
                    missing_indices = torch.nonzero(missing_mask[:, v]).squeeze(1)
                    if len(missing_indices) > 0:
                        view_energies[missing_indices, v] = 1e6
            else:
                view_energies[:, v] = energy_net(features[v]).squeeze(-1)

        # Energy difference calculation
        min_energies, _ = torch.min(view_energies, dim=1, keepdim=True)
        energy_diffs = view_energies - min_energies

        # Convert energy differences to weights
        weights = torch.exp(-energy_diffs / self.temperature)

        # Handle missing views
        if missing_mask is not None:
            weights = weights * (1 - missing_mask.float())

        # Normalize weights
        row_sums = weights.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-10)  # Prevent division by zero
        energy_weights = weights / row_sums

        return energy_weights

    def forward(self, rs, energy_net, missing_mask=None):
        """
        Graph encoding followed by energy-weighted fusion of the refined features over the
        available views, i.e. the reference representation H~ of Eq. (13). Used in the
        autoencoder pretraining stage ('pretrain' mode of ERLNet).

        Returns:
            H: Fused representation [batch_size, feature_dim]
            enhanced_features: GCN-refined view features [n_views][batch_size, feature_dim]
            adj_matrix: Adjacency matrices A_i
            energy_weights: Fusion weights [batch_size, n_views]
        """
        enhanced_features, adj_matrix = self.view_gcn(rs, energy_net, missing_mask)
        energy_weights = self._energy_weights(enhanced_features, energy_net, missing_mask)

        stacked_features = torch.stack(enhanced_features, dim=1)
        weights_expanded = energy_weights.unsqueeze(2)
        H = torch.sum(weights_expanded * stacked_features, dim=1)
        H = normalize(H, dim=1)

        return H, enhanced_features, adj_matrix, energy_weights
