# 4. The model

## The labelling problem

The right training label would be: *when provider X failed, which firms were affected and by
how much.* No such dataset exists publicly. Incident records say an outage happened; they do
not say which downstream firms lost money, and no public source links provider failures to
firm-level impact systematically. Recovering it would require both the incident record and
the dependency structure — and the dependency structure is what this project had to build in
the first place.

Waiting for that data means never building the tool.

## Weak supervision

So the rule-based `CompositeRisk` from [step 6](03-risk-formulas.md#composite-risk) becomes a
**proxy label** over 2022–2024, and the GNN learns the structural pattern that produces it.

The obvious objection: if the label comes from a formula, why not just run the formula?

Because the formula needs the *whole graph* to evaluate. It computes concentration and
2-hop cascades over observed edges — so for any node whose upstream structure is incomplete,
or for a graph year that has changed shape, it either cannot be computed or is computed on
partial information. A trained model infers exposure from *feature and topology patterns*,
so it generalises to nodes and configurations the rule cannot evaluate cleanly. It also
compresses a graph-wide computation into a forward pass, which matters when the intended use
is scoring a new applicant at quote time.

**The honest limitation:** the model can only be as right as the rule. What it adds is
generalisation, not correctness. Nothing here is validated against realised losses, and
every output is a structural-exposure proxy — not a forecast of failure. That caveat is
printed in the generated reports too, because it is the thing most likely to be forgotten
when someone reads a ranked table.

## Setup

- **Train:** 2022–2024 snapshots with weak labels
- **Predict:** 2025 — a graph the model has never seen. Its rule score exists but is never
  shown to the model as a target or a feature; it is kept only so the two can be compared.
- **Features (8):** 6 intrinsic 10-K risks + `incident_signal` + `node_type_enc`
- **Excluded from features:** `pagerank`, `indegree_centrality`, `node_vulnerability`, and
  all four exposure components

That exclusion list is the load-bearing methodological choice. A tabular baseline handed
`pagerank` is not graph-free, and a GNN handed `direct_exposure` is copying its own label
back out. Without it the comparison below would prove nothing.

- **Validation:** 5-fold CV over node-year observations (not over years — holding out a whole
  year would confound structural generalisation with year-over-year graph change)
- **Loss:** MSE masked to labelled nodes. Unlabelled nodes stay in the graph and keep passing
  messages; removing them would change the topology the model is being asked to learn.

## The ablation

| Model | Graph? | Spearman ρ | MAE | RMSE |
|---|---|---|---|---|
| Ridge (tabular) | no | **−0.1375** | 0.1908 | 0.2179 |
| MLP (tabular) | no | 0.1750 | 0.1896 | 0.2140 |
| **GCN** | **yes** | **0.8522** | **0.0717** | **0.0971** |

Reproduce: `python -m src.gnn.cross_val`

### Ridge is negative, and that is the finding

A negative rank correlation is not weak performance — it is *inverted* performance. The firms
whose own filings sound most alarming are, if anything, not the ones the network actually
exposes. Self-reported intrinsic risk is close to uninformative about systemic exposure, and
a linear model built on it ranks firms backwards.

This is the empirical version of the argument in [`01-problem.md`](01-problem.md): an
underwriting process that assesses each firm on its own attributes is not merely missing
some exposure. It may be pointed the wrong way.

### The MLP is what makes the result readable

Comparing only Ridge and the GCN would leave an easy objection: *maybe the relationship is
just non-linear.* The MLP takes exactly the same eight columns, has comparable capacity, and
has no graph — and reaches ρ = 0.175. Non-linearity alone recovers very little.

So the jump to 0.852 is attributable to structure: which nodes sit upstream, how strongly
they connect, and what their neighbours look like. MAE improves 62% over the MLP as well, so
the ranking gain is not bought at the cost of calibration.

### Why GCN and not GAT

A GAT with attention and full edge features was implemented
([`src/gnn/models.py`](../src/gnn/models.py)) and did not beat the GCN by enough to justify
it. In an insurance context the score has to be explainable to an underwriter and defensible
to a regulator, so a near-tie goes to the simpler, more inspectable model.

Two convolution layers, not more — matching the risk formulation, which claims direct and
2-hop cascading exposure and nothing further. A deeper stack would propagate information
further than the underlying definition supports.

## 2025 results

The final GCN, retrained on all 2022–2024 labels, scores **82 nodes** in the 2025 graph.

Top focal firms by predicted systemic exposure: **Amazon (0.651), Alibaba (0.648),
IBM (0.631), Microsoft (0.620), SAP (0.618), Oracle (0.612)**.

These are large platform, cloud and enterprise software firms — but not because the model
rewards size, which it cannot observe: revenue, headcount and market cap are not in the
feature set. They rank high because of *position*. A large ecosystem firm sits inside dense
cross-dependency: subsidiaries, AI partners, cloud relationships, platform integrations.
Each connection is a channel in and a channel out.

Rank agreement between the model and the rule on 2025 is ρ = 0.893 — close enough to confirm
the model learned the intended concept, loose enough that it is not merely recomputing the
formula. The divergences are listed in
`outputs/interpretation_report/top_prediction_divergence_cases.csv`, and they are where a
human underwriter's attention is worth most.

**Underwriting read-through:** for a high-scoring firm, security posture is the wrong
question. Ask instead about cloud redundancy, dependence on specific APIs, whether an
alternative provider is contracted, and what vendor resilience documentation exists.

---

Next: [5. Outputs](05-outputs.md)
