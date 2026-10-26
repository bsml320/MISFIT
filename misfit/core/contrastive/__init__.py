"""Contrastive learning modules for FIRE-MECHANIST.

Pieces: pair construction (build_cross_env_pairs builds same-subject,
cross-environment positives), losses (NT-Xent and supervised contrastive),
hard-negative mining, and the ProjectionHead applied before the loss.
"""

from misfit.core.contrastive.losses import operator_contrastive_loss, supervised_contrastive_loss
from misfit.core.contrastive.mining import mine_hard_negatives
from misfit.core.contrastive.pairs import ContrastiveBatch, build_cross_env_pairs
from misfit.core.contrastive.projection import ProjectionHead

__all__ = [
    "ContrastiveBatch",
    "ProjectionHead",
    "build_cross_env_pairs",
    "mine_hard_negatives",
    "operator_contrastive_loss",
    "supervised_contrastive_loss",
]
