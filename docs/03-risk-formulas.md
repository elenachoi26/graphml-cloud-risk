# 3. Risk formulas

Every weight below lives in [`src/config.py`](../src/config.py) and nowhere else. If a number
here disagrees with that file, the file is right and this document is stale.

---

## Edge weight — how much risk flows down a relationship

```
edge_weight = max(FLOOR, w_css·CSS_norm + w_rls·RLS_norm + w_noalt·NoAlt + w_inv·Inv)

  w_css = 0.40    contract_scale_score, min-max normalised within year
  w_rls = 0.30    risk_language_score, min-max normalised within year
  w_noalt = 0.20  no_alternative_mentioned == "Y"   (binary)
  w_inv = 0.10    investment_mentioned == "Y"       (binary)
  FLOOR = 0.10
```

**Why these weights.** Contract scale leads because disclosed commitment size is the hardest
signal in the set — it is audited, and a firm does not disclose a multi-billion-dollar
commitment to a vendor it could replace. Risk language is next: how alarmed a filing sounds
about a counterparty is informative, but it is rhetoric, and rhetoric varies by legal
counsel more than by exposure. Sole-source language is a strong signal but low-frequency,
so it earns weight without being able to dominate. Investment is last — an equity stake
indicates strategic entanglement, not operational dependence.

**Why a floor.** A firm choosing to name a specific counterparty in a regulatory filing has
already told you something. No disclosed edge should carry zero risk.

Implementation: [`src/step3_edge_weights.py`](../src/step3_edge_weights.py)

---

## Node vulnerability — how bad is it when *this* node breaks

```
NetworkCriticality = 0.50·pagerank + 0.50·indegree_centrality       (both min-max normalised)

NodeVulnerability  = 0.50·IntrinsicRisk
                   + 0.25·IncidentSignal
                   + 0.25·NetworkCriticality
```

where

```
IntrinsicRisk(firm)  = mean of 6 10-K risk features         (0.0 for target entities)
IncidentSignal(firm) = minmax( Σ_events severity × reach × novelty )
```

**Three terms, three sources of evidence.** What the firm admits about itself, what has
actually happened to it, and how many others sit downstream. A node needs only one to
matter — which is what allows a target entity with no filing at all to score high on
structure alone.

**Why pagerank and indegree together.** Indegree counts direct dependents; pagerank weights
them by *their* importance. Being depended on by one hyperscaler is not the same as by five
startups, and indegree alone cannot tell them apart. Equal weights because they are
complementary, not because one was measured to be better.

**The multiplication in IncidentSignal.** severity × reach × novelty, not a sum: a severe
incident nobody noticed and a trivial one that made global news are both less informative
than an event that is severe, wide-reaching and unprecedented. Summing lets a long tail of
minor events outweigh one genuine crisis.

Implementation: [`src/step4_node_vulnerability.py`](../src/step4_node_vulnerability.py)

---

## Four exposure components

A single score is not underwritable — an underwriter must be able to say *why* it is high
and *what would lower it*. So exposure decomposes into four quantities with four different
remedies.

### Direct exposure

```
DirectExposure(i) = Σ_{u ∈ upstream(i)}  edge_weight(u→i) × NodeVulnerability(u)
```

Who you depend on, weighted by how badly. **Remedy:** reduce dependence, or pick a less
fragile provider.

### Cascading exposure

```
CascadingExposure(i) = Σ_{m ∈ upstream(i)} Σ_{u ∈ upstream(m)}
                          NodeVulnerability(u) × edge_weight(u→m) × edge_weight(m→i)
```

Your provider's provider. This is the term that never appears in your own filing, because
you have no relationship with that entity and may not know it exists. Stops at 2 hops —
beyond that the attenuated product stops being distinguishable from noise, and the GNN uses
two convolution layers to match.

**Remedy:** demand upstream transparency from your vendors.

### Dependency concentration

```
DependencyConcentration(i) = Σ_g  s(i,g)²         where s(i,g) = share of i's total
                                                   upstream weight in ecosystem g
```

A Herfindahl index over **parent ecosystems**. 1.0 means everything routes through one
ecosystem — a single point of failure. Grouping by ecosystem rather than legal entity is
essential: AWS, Bedrock and SageMaker do not fail independently, and counting them as three
providers is the same error as calling three policies in one flood plain diversified.

**Remedy:** multi-cloud.

### Structural non-substitutability

```
StructuralNonSubstitutability(i) = 1 / (1 + |distinct upstream ecosystems|)
```

Recoverability. Three independent ecosystems is a migration path; one is an outage that
lasts exactly as long as the provider says it will. This is the duration term — it governs
how long a loss runs, not whether it starts.

**Remedy:** contract a fallback provider, even unused.

Implementation: [`src/step5_risk_components.py`](../src/step5_risk_components.py)

---

## Composite risk

```
CompositeRisk = 0.50·Direct_norm + 0.20·Cascading_norm
              + 0.20·Concentration_norm + 0.10·NonSub_norm
```

Each component is **percentile-rank normalised within its year** before blending.

**Why rank-normalise.** The four raw quantities are on incompatible scales — a sum of
weighted vulnerabilities, a Herfindahl share bounded at 1, a reciprocal count. Blending them
raw would let whichever has the widest range silently dominate. Ranking also keeps scores
comparable across years even as the graph grows from 60 nodes to 82.

**Why 0.50 on direct exposure.** It is the component an underwriter can act on at renewal,
and the one supported by the most direct evidence. The structural terms modulate it: two
firms with identical direct exposure are priced apart by concentration and substitutability.

**What the weights are not.** They are not fitted. There is no ground truth to fit them
against — that is the whole reason for weak supervision. They encode a judgment about what
matters in underwriting, and they are all in one file precisely so a reader can disagree and
re-run: `python run_pipeline.py --stage graph --dry-run` recomputes every distribution and
writes nothing.

Implementation: [`src/step6_composite_risk.py`](../src/step6_composite_risk.py)

---

## Scenario propagation

```
path_score = severity × Π edge_weight(along path) × DECAY^(hop−1)

  DECAY = 0.85, max 3 hops, simple paths only (no node repeats within a path)
```

A node's total exposure to a shock is the **sum over every path** that reaches it. A firm
wired to a provider both directly and through two intermediaries is more exposed than one
with a single link, and a shortest-path view would score them identically.

`DECAY` is a per-hop penalty on top of the multiplicative edge attenuation: distance buys
time. A firm three hops from an outage learns about it later and has more room to fail over
than one wired directly in.

Implementation: [`src/reporting/scenario_propagation.py`](../src/reporting/scenario_propagation.py)

---

## Expected loss (design, not implemented)

The framework stops at exposure. Converting exposure to currency needs insurer-internal
data this project never had:

```
EL(i,j) = p(i,j) × T(j) × L(i) × s(i,j)

  p(i,j)  probability the failure at provider j propagates to firm i
  T(j)    outage duration at j
  L(i)    firm i's loss per hour
  s(i,j)  damage intensity
```

The graph supplies `p` and `s` — propagation probability and damage intensity are what
path scores and edge weights measure. `T` needs provider outage histories; `L` needs sums
insured, coverage limits, deductibles and industry business-interruption sensitivities.

This is stated as a gap rather than filled with estimates: a plausible-looking loss number
built on invented exposure values would be worse than no number, because someone would use
it.
