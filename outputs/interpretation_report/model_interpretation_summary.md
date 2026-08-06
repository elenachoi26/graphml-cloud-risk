# Model Interpretation Summary

## One-Page Summary

**Dataset**: 2022–2025 · 230 node-year observations (train) · 82 nodes scored for 2025
**Graph**: directed, source_firm + target_entity in one node space, dependency edges only (upstream → downstream)
**Weak labels**: CompositeRisk = 0.50×Direct Exposure + 0.20×Cascading Exposure + 0.20×Dependency Concentration + 0.10×Non-Substitutability

| Model | Graph? | Spearman ρ | Note |
|---|---|---|---|
| Ridge (tabular) | no | -0.138 | negative — intrinsic features mis-rank systemic exposure |
| MLP (tabular) | no | 0.175 | non-linearity alone does not close the gap |
| GCN | yes | 0.852 | selected model |

**Key finding**: message passing over the upstream dependency network is the dominant information source for systemic exposure. Node-level tabular features alone are not just insufficient — in the linear case they are misleading.

**Caution**: all outputs are weakly supervised proxies. Predicted risk is not a forecast of firm failure or incident occurrence.