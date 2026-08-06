# Schema — `self_features_mapped.csv`

Firm-level risk features extracted from 10-K filings by an LLM, one row per
(firm, year, feature). Long format rather than wide, because coverage is uneven:
a filing that never discusses energy infrastructure produces no row, which is
different from producing a zero.

**Real panel:** 12 focal firms × 2022–2025 = 48 firm-year observations × 21 features.
**Sample:** `data/samples/self_features_mapped.csv` — 2 fictional firms, 6 features.

## Key columns

| Column | Meaning |
|---|---|
| `Firm_Name` | firm as written in the filing |
| `Year` | fiscal year of the filing |
| `Feature` | which variable this row carries (see below) |
| `__` | the extracted value |
| `Feature Type` | `Risk` or `Financial` |
| `firm_canonical_firm_id` | resolved node id — the join key into the graph |
| `firm_canonical_name`, `firm_entity_type`, `firm_entity_level` | resolved entity attributes |
| `firm_parent_firm_id`, `firm_parent_canonical_name` | parent ecosystem |
| `firm_mapping_found` | whether entity resolution succeeded |

## Variables

Risk intensities are scored on [0, 1] — how strongly the filing signals that risk, not
whether it is mentioned. Presence is nearly constant across filings; intensity is not.

| # | Variable | Meaning |
|---|---|---|
| 1 | `firm_name` | firm (node id) |
| 2 | `vendor_concentration_risk` | dependence on specific suppliers or components |
| 3 | `customer_concentration_risk` | revenue concentration in top customers |
| 4 | `geographic_concentration_risk` | concentration in specific countries or regions |
| 5 | `regulatory_risk` | regulatory, litigation and compliance exposure |
| 6 | `cybersecurity_risk` | cyberattack and data-breach exposure |
| 7 | `operational_continuity_risk` | service interruption and infrastructure failure |
| 8 | `ai_dependency_risk` | dependence on specific AI partners or models |
| 9 | `geopolitical_risk` | geopolitical conflict, trade restrictions, sanctions |
| 10 | `tax_litigation_risk` | tax litigation and transfer-pricing exposure |
| 11 | `energy_infra_risk` | data-centre power and water supply |
| 12 | `financial_stability` | liquidity and leverage concern (0 = stable, 1 = distressed) |
| 13 | `revenue_growth_sentiment` | revenue direction (−1 = declining, 1 = growing) |
| 14 | `debt_to_equity` | total debt / equity |
| 15 | `operating_margin` | operating income / revenue |
| 16 | `rd_intensity` | R&D expense / revenue |
| 17 | `single_region_dependency` | reliance on a single cloud region |
| 18 | `foundation_model_api_dependency` | direct dependence on external AI APIs |
| 19 | `realtime_latency_requirement` | how latency-sensitive the business is |
| 20 | `sensitive_data_handling` | sensitive-data processing exposure |
| 21 | `num_external_api_dependencies` | count of external APIs and services named |

## Which of these reach the model

Only six — `cybersecurity_risk`, `operational_continuity_risk`, `ai_dependency_risk`,
`regulatory_risk`, `geopolitical_risk`, `energy_infra_risk` — are averaged into
`IntrinsicRisk` (`src/config.INTRINSIC_FEATURES`) and passed to the GNN.

The rest are collected and deliberately not fed in. They are firm characteristics rather
than infrastructure-dependency signals, and the whole claim under test is that systemic
exposure comes from network position, not firm attributes. Loading the feature vector with
financials would have made that claim untestable.

Target entities have no filing of their own, so they carry no values here. They are scored
entirely by structure and incident history — which is the point: the nodes that accumulate
the most risk are usually the ones nobody underwrites directly.
