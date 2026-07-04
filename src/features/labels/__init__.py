"""labels — ML target label generators.

Labels represent future price outcomes and are intentionally look-ahead.
They must NEVER be used as input features — only as training targets.
The pipeline validator exempts 'labels' category columns from the
look-ahead naming-pattern check.

Planned generators
------------------
- TripleBarrierLabels: Lopez de Prado triple-barrier method
    Outputs: tb_label (-1 short, 0 neutral, +1 long)
    Uses: ATR-scaled vertical barrier, configurable horizontal barriers
- BinaryDirectionLabel: next N-bar price direction
    Outputs: direction_label_N (0 down, 1 up)
- RiskRewardLabel: label only when minimum RR is achievable
    Outputs: rr_label (filtered by min_rr threshold)
- ContinuousReturnLabel: raw log-return over N bars (regression target)
    Outputs: log_return_N

Each future generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder import LabelsPlaceholder

__all__ = ["LabelsPlaceholder"]
