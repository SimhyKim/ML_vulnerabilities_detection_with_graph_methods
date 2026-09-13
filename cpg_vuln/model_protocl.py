
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, GlobalAttention

from cpg_vuln.domain_knowledge import DOMAIN_GRAPH_DIM
from cpg_vuln.motifs import MOTIF_DIM


class VulProtoCL(nn.Module):
    def __init__(
        self,
        num_type_buckets: int,
        num_token_buckets: int,
        num_edge_types: int = 15,
        embed_dim: int = 64,
        hidden_dim: int = 160,
        num_layers: int = 3,
        num_prototypes: int = 4,
        node_flag_dim: int = 15,
        dropout: float = 0.15,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_prototypes = num_prototypes

        self.type_emb = nn.Embedding(num_type_buckets, embed_dim)
        self.token_emb = nn.Embedding(num_token_buckets, embed_dim)
        self.flag_proj = nn.Linear(node_flag_dim, embed_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_edge_types))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.att_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim * 2) * 0.05)

        self.domain_proj = nn.Linear(DOMAIN_GRAPH_DIM, hidden_dim // 2)
        self.motif_proj = nn.Linear(MOTIF_DIM, hidden_dim // 2)

        head_in = hidden_dim * 2 + hidden_dim
        self.bin_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def encode_graph(self, data) -> torch.Tensor:
        x = self.type_emb(data.type_ids) + self.token_emb(data.token_ids) + self.flag_proj(data.flags)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, data.edge_index, data.edge_type)
            h = self.dropout(self.act(norm(h)))
            x = x + h if h.shape == x.shape else h
        g = torch.cat([global_mean_pool(x, data.batch), self.att_pool(x, data.batch)], dim=-1)
        return g

    def _side_feats(self, data, batch_size: int, device) -> torch.Tensor:
        if hasattr(data, "domain"):
            dom = data.domain
            if dom.dim() == 1:
                dom = dom.view(1, -1)
            if dom.size(0) != batch_size and dom.numel() == batch_size * DOMAIN_GRAPH_DIM:
                dom = dom.view(batch_size, DOMAIN_GRAPH_DIM)
        else:
            dom = torch.zeros(batch_size, DOMAIN_GRAPH_DIM, device=device)

        if hasattr(data, "motif"):
            mot = data.motif
            if mot.dim() == 1:
                mot = mot.view(1, -1)
            if mot.size(0) != batch_size and mot.numel() == batch_size * MOTIF_DIM:
                mot = mot.view(batch_size, MOTIF_DIM)
        else:
            mot = torch.zeros(batch_size, MOTIF_DIM, device=device)

        return torch.cat([self.domain_proj(dom), self.motif_proj(mot)], dim=-1)

    def forward(self, data):
        g = self.encode_graph(data)
        side = self._side_feats(data, g.size(0), g.device)
        feat = torch.cat([g, side], dim=-1)
        logit = self.bin_head(feat).squeeze(-1)
        z = F.normalize(self.proj(g), dim=-1)
        return g, logit, z

    def prototype_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        proto = F.normalize(self.prototypes, dim=-1)
        sim = z @ proto.t() / self.temperature
        vul_mask = y > 0.5
        losses = []
        if vul_mask.any():
            with torch.no_grad():
                nn_idx = sim[vul_mask].argmax(dim=-1)
            losses.append(F.cross_entropy(sim[vul_mask], nn_idx))
        if (~vul_mask).any():
            safe_sim = sim[~vul_mask]
            losses.append(F.softplus(safe_sim.max(dim=-1).values).mean())
        if not losses:
            return z.new_zeros(())
        return sum(losses) / len(losses)

    @staticmethod
    def view_consistency_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()
