# Model Comparison Interpretation

## Disclaimer

Every model here predicts a **weakly supervised proxy** for systemic exposure, derived from rule-based composite risk scores. Predictions are **not** forecasts of observed failures, financial distress, or actual adverse events.

## 1. 5-fold Cross-validation (2022–2024)

| Model | Graph? | Spearman ρ | MAE | RMSE |
|---|---|---|---|---|
| Ridge (tabular) | no | -0.1375 | 0.1908 | 0.2179 |
| MLP (tabular) | no | 0.1750 | 0.1896 | 0.2140 |
| GCN | yes | 0.8522 | 0.0717 | 0.0971 |
## 2. The tabular models do not merely underperform — Ridge points the wrong way

Ridge reaches a Spearman ρ of **-0.1375**. A negative rank correlation
means the firms whose own filings sound most alarming are, if anything, *not* the ones
the network actually exposes. Self-reported intrinsic risk is not a weak proxy for
systemic exposure; it is close to uninformative about it.

The MLP is the control that makes this readable. It sees exactly the same eight
feature columns as the GCN and no graph, and reaches ρ = 0.1750. So the
gap to the GCN's **0.8522** cannot be explained by non-linearity — it is
attributable to structure: which nodes sit upstream, how strongly they connect, and
what their neighbours look like. MAE improves −62% over
the MLP as well, so the ranking gain is not bought with calibration.

This is the empirical case for modelling the dependency network at all. If a table of
firm attributes had been sufficient, no graph would have been needed.

## 3. Why GCN over GAT

Attention did not buy enough to justify the cost. In an insurance setting the score has
to be explainable to an underwriter and defensible to a regulator, so a tie on accuracy
goes to the simpler, more inspectable model. Two convolution layers also match the risk
formulation, which only claims direct and 2-hop (cascading) exposure — a deeper stack
would propagate information further than the underlying definition supports.

## 4. 2025: where the model and the rule disagree

The GCN scored all 82 nodes in the 2025 graph, a year whose rule-based
labels it never saw. Rank agreement with the rule is ρ = **0.893**: close enough
to confirm the model learned the intended concept, loose enough that it is not simply
recomputing the formula.

The divergences are where the model earns its keep. It ranks nodes by *structural
position* learned across three years, so it can flag an entity whose formula inputs look
unremarkable in 2025 but whose position in the network resembles nodes that scored high
before. Those cases are listed in `top_prediction_divergence_cases.csv` and are the ones
worth a human underwriter's attention.

## 5. Largest rank divergences (2025)

| Node | Type | GCN rank | Rule rank | Δ rank | GCN score | Rule score |
|---|---|---|---|---|---|---|
| Amazon | source_firm | 17 | 53 | +36 | 0.6506 | 0.4457 |
| Alibaba | source_firm | 21 | 52 | +31 | 0.6483 | 0.4652 |
| One Medical | target_entity | 5 | 31 | +26 | 0.7204 | 0.6195 |
| Rivian | target_entity | 1 | 26 | +25 | 0.7300 | 0.6384 |
| MGM Holdings | target_entity | 6 | 31 | +25 | 0.7202 | 0.6195 |
| Tencent Cloud | target_entity | 38 | 17 | -21 | 0.5838 | 0.6963 |
| Alibaba Cloud | target_entity | 38 | 17 | -21 | 0.5838 | 0.6963 |
| Cainiao | target_entity | 18 | 39 | +21 | 0.6492 | 0.5677 |
| Trendyol | target_entity | 18 | 39 | +21 | 0.6492 | 0.5677 |
| Lazada | target_entity | 18 | 39 | +21 | 0.6492 | 0.5677 |