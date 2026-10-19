"""Building blocks of ERL-Net.

* ``EnergyFunction``     -- the shared energy function E_theta of Sec. III-C1.
* ``GlobalToViewMapper`` -- the view-specific global-to-view mappers Psi_v of
                            Sec. III-D, Eq. (14).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


class EnergyFunction(nn.Module):
    def __init__(self, r_dim, hidden_dim=256):
        """
        Shared energy function E_theta (Sec. III-C1).

        A single MLP with a SoftPlus output, shared by all views, that assigns a
        non-negative energy score to a feature in the common space R^{d_r}.
        Lower energy means higher reliability.

        Args:
            r_dim: Dimension d_r of the common feature space
            hidden_dim: Hidden layer dimension
        """
        super(EnergyFunction, self).__init__()
        self.r_dim = r_dim

        # MLP with SoftPlus output so that E_theta(r) > 0
        self.energy_net = nn.Sequential(
            nn.Linear(r_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # Apply Kaiming initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        Compute the energy score E_theta(x).

        Args:
            x: Features [batch_size, d_r]

        Returns:
            energy: Energy scores [batch_size, 1]
        """
        # Ensure x is a tensor not a tuple
        if isinstance(x, tuple):
            x = x[0]

        # Calculate energy
        energy = self.energy_net(x)

        return energy


class GlobalToViewMapper(nn.Module):
    def __init__(self, global_dim, view_dims, hidden_dim=256, dropout=0.1):
        """
        Global-to-view mappers Psi_v (Sec. III-D, Eq. (14)).

        One MLP per view that maps a global representation to an imputed
        feature candidate r_hat^v in the common space of view v.

        Args:
            global_dim: Dimension of the global representation (d_r)
            view_dims: List of output dimensions, one per view (d_r for every view)
            hidden_dim: Hidden layer dimension
            dropout: Dropout ratio
        """
        super(GlobalToViewMapper, self).__init__()
        self.num_views = len(view_dims)

        # One mapping network Psi_v per view
        self.mapping_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            ) for dim in view_dims
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, H, view_idx=None):
        """
        Map a global representation to view-specific feature candidates.

        Args:
            H: Global representation [batch_size, global_dim]
            view_idx: Target view index; if None, candidates for all views are returned

        Returns:
            L2-normalized candidate for one view, or a list with one candidate per view
        """
        if view_idx is not None:
            # Map to specific view feature
            view_feature = self.mapping_networks[view_idx](H)
            # Normalization
            return normalize(view_feature, dim=1)
        else:
            # Map to all view features
            view_features = []
            for v in range(self.num_views):
                view_feature = self.mapping_networks[v](H)
                view_features.append(normalize(view_feature, dim=1))
            return view_features
