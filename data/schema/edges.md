# Schema — `edges_risk_propagation_valid.csv`

One row per disclosed relationship between a filing firm and a counterparty it names.
This is the table that turns narrative disclosure into a graph.

**Real panel:** 536 relationship observations across 2022–2025.
**Sample:** `data/samples/edges_risk_propagation_valid.csv` — 5 relationships × 4 years.

## Extracted columns

The LLM reads a filing and, for each counterparty mentioned, emits:

| # | Column | Meaning |
|---|---|---|
| 1 | `Source 기업명` | the filing firm (the one describing the relationship) |
| 2 | `Target 기업명` | the counterparty it names |
| 3 | `relation_type` | nature of the relationship (see below) |
| 4 | `dependency_direction` | `upstream` / `downstream` — which way the dependency runs |
| 5 | `risk_language_score` | [0,1] how alarmed the filing's own wording is |
| 6 | `investment_mentioned` | `Y`/`N` — equity stake or strategic investment disclosed |
| 7 | `contract_scale_score` | [0,1] disclosed contract or commitment size |
| 8 | `no_alternative_mentioned` | `Y`/`N` — "sole source", "no comparable alternative" |
| 9 | `spillover_type` | how risk would travel (see below) |
| 10 | `source_section` | filing section the sentence came from (`Item1A`, `Item7`, …) |
| — | `근거 문장` | the verbatim sentence. **Stripped from all published CSVs.** |

## Resolved columns

Both endpoints are resolved to canonical ids, with parent ecosystem attached:
`source_canonical_firm_id`, `target_canonical_firm_id`, `*_entity_type`, `*_entity_level`,
`*_parent_firm_id`, `*_mapping_found`.

## Direction columns — the important part

`Source`/`Target` describe *who mentioned whom*, which is a documentation artifact, not a
risk direction. A firm's 10-K names its suppliers, so the filer is usually the one exposed.

`risk_source_id → risk_target_id` is the **propagation** direction: upstream origin →
downstream exposed node. `risk_direction_rule` records how it was derived
(`target_is_upstream_of_source` / `source_is_upstream_of_target`), and `risk_edge_valid`
marks rows where a direction could be established. Everything downstream of this table —
weights, exposure components, centrality, scenarios — uses the risk direction, never the
mention direction.

Getting this backwards inverts the entire model, so it is stored explicitly rather than
inferred at read time.

## `relation_type`

`cloud_provider`, `ai_provider`, `software_vendor`, `supplier`, `partner`, `subsidiary`,
`investment_target`, `customer`, and others.

Four are dropped before the graph is built (`src/config.REMOVE_RELATIONS`):
`competitor`, `regulator`, `investor`, `investee`. They are real relationships that carry no
operational-failure semantics — a competitor's outage does not interrupt your service, and
leaving them in would inflate centrality for well-connected firms that are not actually
exposed to each other.

## `spillover_type`

How risk travels, given that the dependency exists:

| Type | Meaning | Example |
|---|---|---|
| `operational` | provider failure → downstream service interruption | cloud region outage, API down |
| `security` | vulnerability or breach spreads to connected firms | provider breach, compromised auth service |
| `performance` | latency or quality degradation reaches customers | slow model responses, degraded API |
| `governance` | policy, access or regulatory change alters operations | model access restricted, terms change |

The same dependency can carry different risk depending on channel, which is why the type is
recorded per edge rather than inferred from `relation_type`. It is also what lets an insurer
line up a scenario with a specific coverage: `operational` maps to contingent business
interruption, `security` to a cyber policy, `governance` to neither.

## Edge weight

`src/step3_edge_weights.py` combines four of these into a single [0.1, 1] weight:

```
edge_weight = max(0.10, 0.40·contract_scale + 0.30·risk_language
                      + 0.20·no_alternative + 0.10·investment)
```

Contract scale and risk language are min-max normalised within the year first.
See `docs/03-risk-formulas.md`.
